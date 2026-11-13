import argparse
import json
from pathlib import Path
import random
import os
import schedulefree

import numpy as np
import torch
import wandb

import config
from data.utils import DataReader, get_dataset
import distributed
from models.utils import get_model
from optim.base import train
from optim.utils import cos_inf_schedule, wsd_schedule
from optim.muon import Muon
from optim.muon_gd_ada import MuonGDAda


def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.train_data_seed is None:
        args.train_data_seed = args.data_seed
    if args.eval_data_seed is None:
        args.eval_data_seed = args.data_seed

    if args.full_eval_at is None:
        args.full_eval_at = []
    if args.permanent_ckpt_at is None:
        args.permanent_ckpt_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    exp_name = get_exp_name(args, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=vars(args),
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("train_eval/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(args.device)
    # TODO: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {"parameters": params_cnt, "optimized_parameters": optimized_params_cnt}
        )

    if args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "SFAdamW":
        opt = schedulefree.AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )
    elif args.opt == "muon":
        opt = Muon(
            group_specs,
            lr=args.lr,
            momentum=args.muon_gd_momentum,
        )
    elif args.opt == "muon_gd_ada":
        opt = MuonGDAda(
            group_specs,
            lr=args.lr,
            momentum=args.muon_gd_momentum,
            gd_lr=args.gd_lr,
            gd_momentum=args.gd_momentum,
            weight_decay=args.weight_decay,
            psi=args.psi,
            adapt_interval=args.adapt_interval,
            warmup_steps=args.muon_gd_warmup_steps,
            psi_up=args.psi_up,
            psi_down=args.psi_down,
            initial_mode=args.initial_mode,
            ns_steps=args.ns_steps,
            rank_tol=args.rank_tol,
        )

    else:
        opt = torch.optim.SGD(
            group_specs, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay
        )
    print(f"\nOptimizer:\n{opt}")

    # Read resume iteration early so scheduler can be configured correctly when
    # continuing from a checkpoint with a different optimizer.
    resume_iter = 0
    if args.resume_from is not None:
        ckpt_file = Path(args.resume_from) / "main.pt"
        if ckpt_file.exists():
            resume_iter = torch.load(ckpt_file, map_location="cpu")["itr"]

    if args.scheduler != "none":
        if args.scheduler in ["cos", "linear"]:
            if args.resume_from is not None and (not args.load_scheduler_state):
                total_steps = args.iterations - resume_iter
                assert (
                    total_steps > 1
                ), "Remaining steps must be > 1 for resumed OneCycle schedule."
                warmup_steps = min(args.second_warmup_steps, total_steps - 1)
            else:
                total_steps = args.iterations
                warmup_steps = args.warmup_steps
                assert (
                    warmup_steps < total_steps
                ), "Warmup steps must be < iterations."

            # initial lr is args.lr / div_factor
            # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[group.get("lr", args.lr) for group in group_specs],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=1,
                final_div_factor=10,
            )
        elif args.scheduler == "cos_inf":
            if args.resume_from is not None and (not args.load_scheduler_state):
                total_steps = args.iterations - resume_iter
                assert (
                    total_steps > 1
                ), "Remaining steps must be > 1 for resumed cos_inf schedule."
                warmup_steps = min(args.second_warmup_steps, total_steps - 1)
            else:
                total_steps = args.iterations
                warmup_steps = args.warmup_steps

            assert (
                warmup_steps < total_steps
            ), "Warmup steps must be < total cos_inf steps."
            assert (
                0 <= args.cos_inf_steps < total_steps
            ), "cos_inf_steps must be >= 0 and < total cos_inf steps."
            assert (
                warmup_steps < total_steps - args.cos_inf_steps
            ), "Warmup steps must end before the cos_inf constant tail."

            lambda_schedule = cos_inf_schedule(
                n_iterations=total_steps,
                n_warmup=warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1,
                final_div_factor=1000000,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "wsd":
            if args.resume_from is not None and (not args.load_scheduler_state):
                total_steps = args.iterations - resume_iter
                warmup_steps = min(args.second_warmup_steps, total_steps - 1)
            else:
                total_steps = args.iterations
                warmup_steps = args.warmup_steps

            lambda_schedule = wsd_schedule(
                n_iterations=total_steps,
                n_warmup=warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")

    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args, distributed_backend):
    """Returns the name of the experiment, used for saving models and wandb."""
    if args.experiment_name is not None:
        return args.experiment_name

    rank = distributed_backend.rank

    exp_name = (
        f"{args.dataset}_{args.model}_nlayers{args.n_layer}"
        f"_nhead{args.n_head}_lr{args.lr}"
        f"_sched_{args.scheduler}_warmup{args.warmup_steps}"
        f"_decay_{args.decay_type}_{args.wsd_fract_decay}"
        f"_iter{args.iterations}"
        f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
    )
    # for mup
    if args.model == "mup_noam":
        exp_name = (
            f"{args.dataset}_{args.model}"
            f"_opt{args.opt}"
            f"_nlayers{args.n_layer}"
            # f"_nhead{args.n_head}"
            f"_lr{args.lr}"
            f"_sched_{args.scheduler}"
            f"_decay_{args.decay_type}"
            # f"_warmup{args.warmup_steps}"
            f"_iter{args.iterations}"
            f"_init{args.init_std}_sce{args.scale_emb}"
            f"_scd{args.scale_depth}"
            # f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
        )
    if args.wandb_run_prefix != "none":
        exp_name = args.wandb_run_prefix + "_" + exp_name
    exp_name += f"_seed{args.seed - rank}"
    exp_name += f"_train_data_seed{args.train_data_seed}"
    exp_name += f"_eval_data_seed{args.eval_data_seed}"

    if args.weight_average:
        exp_name += f"_WA"
    if args.opt == "SFAdamW":
        exp_name += f"_beta1_{args.beta1}"
        exp_name += f"_beta2_{args.beta2}"
    return exp_name


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.train_data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    # Use a separate reader for fixed-batch train-set evaluation. Resetting this
    # reader never changes the state or batch order of the actual training
    # reader. The fixed evaluation seed also keeps this probe set identical
    # across optimizer runs whose training-data seeds differ.
    train_eval_data_src = (
        train_reader.data if train_reader.data is not None else data_srcs["train"]
    )
    train_eval_reader = DataReader(
        data_src=train_eval_data_src,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.eval_data_seed,
        with_replacement=False,
        auto_shard=False,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.eval_data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "train_eval": train_eval_reader,
        "val": val_reader,
    }


if __name__ == "__main__":
    args = get_args()
    main(args)
