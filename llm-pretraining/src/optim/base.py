from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml

import torch
import wandb

from logger.logger import DynamicsLogger
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)
from .state_transfer import (
    transfer_muon_momentum_to_adamw,
    transfer_muon_momentum_to_sgd,
)


def _mean_across_workers(value, device):
    """Return the mean of one scalar value across distributed workers."""
    if torch.is_tensor(value):
        value_tensor = value.detach().to(device=device, dtype=torch.float64)
    else:
        value_tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.SUM)
        value_tensor /= torch.distributed.get_world_size()
    return value_tensor.item()


def log_online_train_window(
    log_iter,
    epoch,
    loss_sum,
    loss_count,
    iter_time_sum,
    iter_count,
    opt,
    distributed_backend,
    cfg,
):
    """Log the mean online loss over every microbatch since the last log."""
    train_loss = _mean_across_workers(loss_sum / loss_count, cfg.device)
    mean_iter_time = _mean_across_workers(iter_time_sum / iter_count, cfg.device)
    current_lrs = [param_group["lr"] for param_group in opt.param_groups]

    if distributed_backend.is_master_process():
        print(
            f"Train: Iter={log_iter} ({epoch:0.3f} epochs) "
            f"train_loss={train_loss:.3f} iter_dt={mean_iter_time:.2e}s "
            f"lr={current_lrs[0]:.2e}"
        )

        if cfg.wandb:
            wandb.log(
                {
                    "iter": log_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    "iter_dt": mean_iter_time,
                }
            )

    return train_loss


def estimate_adamw_warm_start_stats(
    model,
    opt,
    train_reader,
    substep,
    cfg,
    type_ctx,
    distributed_backend,
):
    if cfg.adamw_v_calib_batches <= 0:
        return None

    was_training = model.training
    model.train()
    opt.zero_grad(set_to_none=True)

    dst_sd = opt.state_dict()
    grad_avg = {}
    grad_sq = {}

    for group in dst_sd["param_groups"]:
        for pid in group["params"]:
            grad_avg[pid] = None
            grad_sq[pid] = None

    for _ in range(cfg.adamw_v_calib_batches):
        x, y = get_batch(train_reader, device=cfg.device)

        with type_ctx:
            outputs = model(x, targets=y)

        loss = outputs["loss"]
        loss.backward()

        for group, sd_group in zip(opt.param_groups, dst_sd["param_groups"]):
            for p, pid in zip(group["params"], sd_group["params"]):
                if p.grad is None:
                    continue

                g = p.grad.detach().float()
                g2 = g.pow(2)

                if grad_avg[pid] is None:
                    grad_avg[pid] = g.clone()
                    grad_sq[pid] = g2.clone()
                else:
                    grad_avg[pid].add_(g)
                    grad_sq[pid].add_(g2)

        opt.zero_grad(set_to_none=True)

    for pid in grad_avg:
        if grad_avg[pid] is not None:
            grad_avg[pid].div_(cfg.adamw_v_calib_batches)
            grad_sq[pid].div_(cfg.adamw_v_calib_batches)

    train_reader.set_step(substep)
    opt.zero_grad(set_to_none=True)

    if not was_training:
        model.eval()

    return {
        "grad_avg": grad_avg,
        "grad_sq": grad_sq,
        "num_batches": cfg.adamw_v_calib_batches,
    }


def estimate_grad_sq_for_adamw_v(
    model,
    opt,
    train_reader,
    substep,
    cfg,
    type_ctx,
    distributed_backend,
):
    if cfg.adamw_v_calib_batches <= 0:
        return None

    was_training = model.training
    model.train()
    opt.zero_grad(set_to_none=True)

    grad_sq = {}
    dst_sd = opt.state_dict()

    for group in dst_sd["param_groups"]:
        for pid in group["params"]:
            grad_sq[pid] = None

    for _ in range(cfg.adamw_v_calib_batches):
        x, y = get_batch(train_reader, device=cfg.device)

        with type_ctx:
            outputs = model(x, targets=y)

        loss = outputs["loss"]
        loss.backward()

        for group, sd_group in zip(opt.param_groups, dst_sd["param_groups"]):
            for p, pid in zip(group["params"], sd_group["params"]):
                if p.grad is None:
                    continue

                g2 = p.grad.detach().float().pow(2)
                if grad_sq[pid] is None:
                    grad_sq[pid] = g2.clone()
                else:
                    grad_sq[pid].add_(g2)

        opt.zero_grad(set_to_none=True)

    for pid, value in grad_sq.items():
        if value is not None:
            value.div_(cfg.adamw_v_calib_batches)

    train_reader.set_step(substep)
    opt.zero_grad(set_to_none=True)

    if not was_training:
        model.eval()

    return grad_sq

def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    # track whether we resumed and the iteration we resumed at
    resume_iter_val = None
    resume_train_loss = None
    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not intended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter, ckpt_meta = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
            load_optimizer_state=cfg.load_optimizer_state,
            load_scheduler_state=cfg.load_scheduler_state,
            return_checkpoint_meta=True,
        )
        resume_train_loss = ckpt_meta.get("last_train_loss", None)
        load_worker_state(ckpt_dir)
        resume_iter_val = curr_iter
    else:
        curr_iter = 0
    substep = curr_iter * cfg.acc_steps
    train_reader = datareaders["train"]
    train_eval_reader = datareaders["train_eval"]
    val_reader = datareaders["val"]
    train_reader.set_step(substep)
    # Optional optimizer-state transfer for optimizer switching.
    # This should happen after model resume and before the training loop.
    transfer_mode = getattr(cfg, "transfer_momentum", "none")

    if transfer_mode == "muon_to_adamw":
        if cfg.opt != "adamw":
            raise ValueError(
                "--transfer-momentum muon_to_adamw requires --opt adamw, "
                f"but got --opt {cfg.opt}"
            )

        if cfg.load_optimizer_state:
            raise ValueError(
                "Do not use --load-optimizer-state together with "
                "--transfer-momentum muon_to_adamw. "
                "Muon optimizer state must be converted, not loaded directly."
            )

        if cfg.resume_from is None and cfg.momentum_source_ckpt is None:
            raise ValueError(
                "--transfer-momentum muon_to_adamw requires either "
                "--resume-from or --momentum-source-ckpt"
            )

        momentum_source_ckpt = cfg.momentum_source_ckpt or cfg.resume_from

        print(
            f"[INFO] Transferring Muon momentum into AdamW from: "
            f"{momentum_source_ckpt}"
        )

        warm_start_stats = None
        if (
            cfg.adamw_exp_avg_sq_init in ["grad_sq_calib", "max_grad_sq_muon_sq"]
            or cfg.adamw_exp_avg_init == "update_space_muon"
        ):
            print(
                f"[INFO] Estimating AdamW warm-start stats from "
                f"{cfg.adamw_v_calib_batches} calibration batches."
            )
            warm_start_stats = estimate_adamw_warm_start_stats(
                model=model,
                opt=opt,
                train_reader=train_reader,
                substep=substep,
                cfg=cfg,
                type_ctx=type_ctx,
                distributed_backend=distributed_backend,
            )

        transfer_stats = transfer_muon_momentum_to_adamw(
            adamw_optimizer=opt,
            ckpt_dir=momentum_source_ckpt,
            exp_avg_init=cfg.adamw_exp_avg_init,
            exp_avg_sq_init=cfg.adamw_exp_avg_sq_init,
            scale=cfg.momentum_transfer_scale,
            warm_start_stats=warm_start_stats,
            grad_sq_scale=cfg.adamw_v_calib_scale,
            grad_sq_floor=cfg.adamw_v_floor,
            exp_avg_sq_scale=cfg.adamw_exp_avg_sq_scale,
            update_blend_alpha=cfg.adamw_update_blend_alpha,
            muon_v_scale=cfg.adamw_muon_v_scale,
            step_init=cfg.adamw_transfer_step_init,
        )

        if distributed_backend.is_master_process() and cfg.wandb:
            wandb.log(
                {
                    "iter": curr_iter,
                    "momentum_transfer/transferred": transfer_stats["transferred"],
                    "momentum_transfer/skipped": transfer_stats["skipped"],
                    "momentum_transfer/source_itr": transfer_stats["source_itr"],
                }
            )

    elif transfer_mode == "muon_to_sgd":
        if cfg.opt != "sgd":
            raise ValueError(
                "--transfer-momentum muon_to_sgd requires --opt sgd, "
                f"but got --opt {cfg.opt}"
            )

        if cfg.load_optimizer_state:
            raise ValueError(
                "Do not use --load-optimizer-state together with "
                "--transfer-momentum muon_to_sgd. "
                "Muon optimizer state must be converted, not loaded directly."
            )

        if cfg.resume_from is None and cfg.momentum_source_ckpt is None:
            raise ValueError(
                "--transfer-momentum muon_to_sgd requires either "
                "--resume-from or --momentum-source-ckpt"
            )

        momentum_source_ckpt = cfg.momentum_source_ckpt or cfg.resume_from

        print(
            f"[INFO] Transferring Muon momentum into SGD from: "
            f"{momentum_source_ckpt}"
        )

        transfer_stats = transfer_muon_momentum_to_sgd(
            sgd_optimizer=opt,
            ckpt_dir=momentum_source_ckpt,
            scale=cfg.momentum_transfer_scale,
        )

        if distributed_backend.is_master_process() and cfg.wandb:
            wandb.log(
                {
                    "iter": curr_iter,
                    "momentum_transfer/transferred": transfer_stats["transferred"],
                    "momentum_transfer/skipped": transfer_stats["skipped"],
                    "momentum_transfer/source_itr": transfer_stats["source_itr"],
                }
            )

    elif transfer_mode != "none":
        raise ValueError(f"Unknown transfer_momentum mode: {transfer_mode}")

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    # Online train/loss is the mean over all training microbatches between two
    # logging points. This avoids reporting one arbitrarily easy or hard batch.
    last_train_loss = None
    train_loss_sum = None
    train_loss_count = 0
    train_iter_time_sum = 0.0
    train_iter_count = 0

    # If we resumed from a checkpoint, run one full evaluation immediately at
    # the resume iteration so that downstream plots (e.g. optimizer A -> B
    # switching) have a contiguous evaluation point at the boundary. We then
    # skip the duplicate eval for that same iteration inside the main loop.
    resume_eval_done = False
    if resume_iter_val is not None:
        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        # Use the same fixed-batch protocol at the resume boundary so the
        # pre- and post-switch curves meet at a directly comparable point.
        eval_and_log(
            curr_iter,
            epoch,
            model,
            train_eval_reader,
            val_reader,
            type_ctx,
            distributed_backend,
            cfg,
            opt,
            full_eval=False,
        )

        # Log resume-boundary train point using the exact value saved in the
        # checkpoint (if present). This prevents a visual jump where the first
        # point in the resumed run would otherwise be a post-update loss from
        # the new optimizer at the same iteration index.
        if (
            distributed_backend.is_master_process()
            and cfg.wandb
            and resume_train_loss is not None
        ):
            current_lrs = [param_group["lr"] for param_group in opt.param_groups]
            wandb.log(
                {
                    "iter": curr_iter,
                    "train/loss": resume_train_loss,
                    "train/perplexity": 2.71828 ** resume_train_loss,
                    "lr": current_lrs[0],
                }
            )
        resume_eval_done = True

    while curr_iter <= cfg.iterations:
        # Flush a partial logging window at the exact final iteration. Doing it
        # before checkpointing also stores the same value shown in the logs.
        if curr_iter == cfg.iterations and train_loss_count > 0:
            ws = distributed_backend.get_world_size()
            tokens = ws * substep * cfg.sequence_length * cfg.batch_size
            epoch = tokens / train_reader.num_tokens
            last_train_loss = log_online_train_window(
                curr_iter,
                epoch,
                train_loss_sum,
                train_loss_count,
                train_iter_time_sum,
                train_iter_count,
                opt,
                distributed_backend,
                cfg,
            )
            train_loss_sum = None
            train_loss_count = 0
            train_iter_time_sum = 0.0
            train_iter_count = 0

        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0 or curr_iter in cfg.permanent_ckpt_at:
            if (
                (cfg.permanent_ckpt_interval > 0 and curr_iter % cfg.permanent_ckpt_interval == 0)
                or curr_iter in cfg.permanent_ckpt_at
            ):
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(
                        model,
                        opt,
                        scheduler,
                        curr_iter,
                        ckpt_dir,
                        last_train_loss=last_train_loss,
                    )
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(
                        model,
                        opt,
                        scheduler,
                        curr_iter,
                        ckpt_dir,
                        last_train_loss=last_train_loss,
                    )
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens

        # Skip evaluation here if we already performed the resume-time eval for
        # the same iteration to avoid duplicate logging (which would still
        # create a visual gap). Otherwise follow the usual eval schedule.
        if (
            (curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at))
            and not (resume_iter_val is not None and resume_eval_done and curr_iter == resume_iter_val)
        ):
            eval_and_log(
                curr_iter,
                epoch,
                model,
                train_eval_reader,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            microbatch_loss = outputs["loss"]
            detached_loss = microbatch_loss.detach().float()
            train_loss_sum = (
                detached_loss
                if train_loss_sum is None
                else train_loss_sum + detached_loss
            )
            train_loss_count += 1
            loss = microbatch_loss / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.opt == "SFAdamW":
            opt.train()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)
        if cfg.weight_average:
            weight_averager.step(not_compiled_model, distributed_backend.is_master_process())
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9
        train_iter_time_sum += dt
        train_iter_count += 1

        completed_iter = curr_iter + 1
        if cfg.log_interval and completed_iter % cfg.log_interval == 0:
            ws = distributed_backend.get_world_size()
            tokens = ws * substep * cfg.sequence_length * cfg.batch_size
            completed_epoch = tokens / train_reader.num_tokens
            last_train_loss = log_online_train_window(
                completed_iter,
                completed_epoch,
                train_loss_sum,
                train_loss_count,
                train_iter_time_sum,
                train_iter_count,
                opt,
                distributed_backend,
                cfg,
            )
            train_loss_sum = None
            train_loss_count = 0
            train_iter_time_sum = 0.0
            train_iter_count = 0

        curr_iter = completed_iter

    return stats


def eval_and_log(
    curr_iter,
    epoch,
    model,
    train_eval_reader,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "SFAdamW":
        opt.eval()

    # Measure train-set and validation loss with the same evaluation protocol:
    # eval mode, no gradients, fixed batches, and the same number of batches.
    # This is intentionally separate from the noisy online train/loss metric.
    train_eval_reader.set_step(0)
    train_eval_acc, train_eval_loss, train_eval_perplexity = eval(
        model,
        train_eval_reader,
        cfg.device,
        max_num_batches=cfg.eval_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    if full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"train_eval_loss={train_eval_loss:.3f} "
        f"train_eval_pp={train_eval_perplexity:.3f} "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )
    if curr_iter == 0:
        # No online training window exists before the first update. Use the
        # fixed train-set evaluation as the well-defined initialization point
        # so train/loss curves share a step-zero baseline.
        print(
            f"Train: Iter=0 ({epoch:0.3f} epochs) "
            f"train_loss={train_eval_loss:.3f} (initial fixed-train evaluation)"
        )

    if cfg.wandb:
        if full_eval:
            logs = {
                "iter": curr_iter,
                "train_eval/loss": train_eval_loss,
                "train_eval/perplexity": train_eval_perplexity,
                "train_eval/acc": train_eval_acc,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "train_eval/loss": train_eval_loss,
                "train_eval/perplexity": train_eval_perplexity,
                "train_eval/acc": train_eval_acc,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        if curr_iter == 0:
            logs.update(
                {
                    "train/loss": train_eval_loss,
                    "train/perplexity": train_eval_perplexity,
                }
            )

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
