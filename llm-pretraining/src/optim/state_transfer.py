import os
import torch
import math
from .opt_utils import zeropower_via_newtonschulz5

def _rms(x, eps=1e-12):
    return x.float().pow(2).mean().sqrt().clamp_min(eps)


def _muon_update_from_momentum(momentum, ns_steps=5):
    if momentum.ndim < 2:
        return momentum

    orig_shape = momentum.shape
    if momentum.ndim > 2:
        m2d = momentum.view(orig_shape[0], -1)
    else:
        m2d = momentum

    update_2d = zeropower_via_newtonschulz5(m2d, steps=ns_steps)

    rows, cols = m2d.size()
    scale = (
        max(1.0, math.sqrt(rows / cols))
        if rows > cols
        else max(1.0, math.sqrt(cols / rows))
    )
    update_2d.mul_(scale)

    return update_2d.reshape(orig_shape)



def transfer_muon_momentum_to_adamw(
    adamw_optimizer,
    ckpt_dir,
    exp_avg_init="muon_momentum",
    exp_avg_sq_init="momentum_sq",
    scale=1.0,
    warm_start_stats=None,
    grad_sq_scale=1.0,
    grad_sq_floor=1e-12,
    exp_avg_sq_scale=1.0,
    update_blend_alpha=0.8,
    muon_v_scale=0.01,
    step_init="source_itr",
):
    """
    Transfer Muon momentum_buffer from a Muon checkpoint into a freshly
    constructed AdamW optimizer.

    Source:
        ckpt_dir/main.pt["optimizer"]["state"][param_id]["momentum_buffer"]

    Target AdamW:
        state[param]["exp_avg"]    <- momentum_buffer
        state[param]["exp_avg_sq"] <- momentum_buffer ** 2 or zeros
        state[param]["step"]       <- checkpoint iteration
    """

    main_path = os.path.join(ckpt_dir, "main.pt")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Cannot find checkpoint file: {main_path}")

    ckpt = torch.load(main_path, map_location="cpu")

    if "optimizer" not in ckpt:
        raise KeyError(f"No optimizer state found in {main_path}. Keys: {ckpt.keys()}")

    if "itr" not in ckpt:
        raise KeyError(f"No iteration key 'itr' found in {main_path}. Keys: {ckpt.keys()}")

    src_optim_sd = ckpt["optimizer"]
    src_state = src_optim_sd["state"]
    src_groups = src_optim_sd["param_groups"]

    dst_sd = adamw_optimizer.state_dict()
    dst_groups = dst_sd["param_groups"]

    new_state = {}
    transferred = 0
    skipped = 0

    if len(src_groups) != len(dst_groups):
        print(
            f"[WARN] Source optimizer has {len(src_groups)} param groups, "
            f"AdamW has {len(dst_groups)} param groups."
        )

    for src_group, dst_group in zip(src_groups, dst_groups):
        src_param_ids = src_group["params"]
        dst_param_ids = dst_group["params"]

        if len(src_param_ids) != len(dst_param_ids):
            print(
                f"[WARN] Param group size mismatch: "
                f"source={len(src_param_ids)}, target={len(dst_param_ids)}"
            )

        for src_pid, dst_pid in zip(src_param_ids, dst_param_ids):
            src_param_state = src_state.get(src_pid, None)

            if src_param_state is None:
                skipped += 1
                continue

            momentum = src_param_state.get("momentum_buffer", None)

            if momentum is None:
                skipped += 1
                continue

            if not torch.is_tensor(momentum):
                skipped += 1
                continue

            momentum = momentum.detach().float() * scale

            adamw_state = {}

            grad_avg = None
            grad_sq = None
            target_device = momentum.device

            if warm_start_stats is not None:
                grad_avg = warm_start_stats.get("grad_avg", {}).get(dst_pid, None)
                grad_sq = warm_start_stats.get("grad_sq", {}).get(dst_pid, None)

                if grad_sq is not None:
                    target_device = grad_sq.device
                elif grad_avg is not None:
                    target_device = grad_avg.device

            momentum = momentum.to(target_device)
            # 1. Build AdamW second moment v = exp_avg_sq first.
            if exp_avg_sq_init == "zeros":
                v = torch.zeros_like(momentum)

            elif exp_avg_sq_init == "momentum_sq":
                v = momentum.pow(2) * exp_avg_sq_scale

            elif exp_avg_sq_init == "grad_sq_calib":
                if grad_sq is None:
                    skipped += 1
                    continue
                v = grad_sq.detach().float() * grad_sq_scale

            elif exp_avg_sq_init == "max_grad_sq_muon_sq":
                if grad_sq is None:
                    skipped += 1
                    continue
                v_grad = grad_sq.detach().float() * grad_sq_scale
                v_muon = momentum.pow(2) * muon_v_scale
                v = torch.maximum(v_grad, v_muon)

            else:
                raise ValueError(f"Unknown exp_avg_sq_init: {exp_avg_sq_init}")

            if grad_sq_floor > 0:
                floor = v.mean().clamp_min(1e-30) * grad_sq_floor
                v = torch.clamp(v, min=floor)

            # 2. Build AdamW first moment m = exp_avg.
            if exp_avg_init == "muon_momentum":
                m = momentum.clone()

            elif exp_avg_init == "update_space_muon":
                sqrt_v = v.sqrt().clamp_min(1e-12)

                u_muon = _muon_update_from_momentum(momentum)

                if grad_avg is not None:
                    g = grad_avg.detach().float()
                    u_grad = g / sqrt_v

                    # Match Muon update-space RMS to Adam gradient update-space RMS.
                    u_muon = u_muon * (_rms(u_grad) / _rms(u_muon))

                    alpha = update_blend_alpha
                    m_update = alpha * u_muon + (1.0 - alpha) * u_grad
                else:
                    m_update = u_muon

                # Choose m so AdamW's preconditioned direction starts near m_update.
                m = m_update * sqrt_v

            else:
                raise ValueError(f"Unknown exp_avg_init: {exp_avg_init}")

            # 3. Build AdamW step.
            if step_init == "source_itr":
                step_value = float(ckpt["itr"])
            elif step_init == "zero":
                step_value = 0.0
            elif step_init == "calib_batches":
                step_value = float(
                    0 if warm_start_stats is None else warm_start_stats.get("num_batches", 0)
                )
            else:
                raise ValueError(f"Unknown step_init: {step_init}")

            adamw_state["step"] = torch.tensor(step_value)
            adamw_state["exp_avg"] = m.clone()
            adamw_state["exp_avg_sq"] = v.clone()

            new_state[dst_pid] = adamw_state
            transferred += 1

    dst_sd["state"] = new_state

    # This maps integer state_dict param IDs back onto the live AdamW parameters.
    adamw_optimizer.load_state_dict(dst_sd)

    print(
        f"[INFO] Muon -> AdamW momentum transfer complete. "
        f"Transferred {transferred} tensors, skipped {skipped}. "
        f"source_itr={ckpt['itr']}, exp_avg_init={exp_avg_init}, "
        f"exp_avg_sq_init={exp_avg_sq_init}, momentum_scale={scale}, "
        f"grad_sq_scale={grad_sq_scale}, grad_sq_floor={grad_sq_floor}, "
        f"exp_avg_sq_scale={exp_avg_sq_scale}, update_blend_alpha={update_blend_alpha}, "
        f"muon_v_scale={muon_v_scale}, step_init={step_init}"
    )

    return {
        "transferred": transferred,
        "skipped": skipped,
        "source_itr": ckpt["itr"],
    }
    
def transfer_muon_momentum_to_sgd(
    sgd_optimizer,
    ckpt_dir,
    scale=1.0,
    convert_ema_to_sgd_buffer=True,
):
    """
    Transfer Muon momentum_buffer from a Muon checkpoint into a freshly
    constructed torch.optim.SGD optimizer with momentum.

    Source:
        ckpt_dir/main.pt["optimizer"]["state"][param_id]["momentum_buffer"]

    Target SGD:
        state[param]["momentum_buffer"] <- Muon momentum_buffer converted to
                                           PyTorch SGD's buffer convention

    Muon stores an EMA-style buffer:
        m_t = beta * m_{t-1} + (1 - beta) * grad_t

    torch.optim.SGD stores a velocity-style buffer:
        v_t = momentum * v_{t-1} + (1 - dampening) * grad_t

    When beta == momentum and dampening == 0, the equivalent SGD buffer is:
        v_t = m_t / (1 - beta)
    """

    import os
    import torch

    main_path = os.path.join(ckpt_dir, "main.pt")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Cannot find checkpoint file: {main_path}")

    ckpt = torch.load(main_path, map_location="cpu")

    if "optimizer" not in ckpt:
        raise KeyError(f"No optimizer state found in {main_path}. Keys: {ckpt.keys()}")

    if "itr" not in ckpt:
        raise KeyError(f"No iteration key 'itr' found in {main_path}. Keys: {ckpt.keys()}")

    src_optim_sd = ckpt["optimizer"]
    src_state = src_optim_sd["state"]
    src_groups = src_optim_sd["param_groups"]

    dst_sd = sgd_optimizer.state_dict()
    dst_groups = dst_sd["param_groups"]

    new_state = {}
    transferred = 0
    skipped = 0

    if len(src_groups) != len(dst_groups):
        print(
            f"[WARN] Source optimizer has {len(src_groups)} param groups, "
            f"SGD has {len(dst_groups)} param groups."
        )

    for src_group, dst_group, live_group in zip(src_groups, dst_groups, sgd_optimizer.param_groups):
        src_param_ids = src_group["params"]
        dst_param_ids = dst_group["params"]
        live_params = live_group["params"]

        if len(src_param_ids) != len(dst_param_ids):
            print(
                f"[WARN] Param group size mismatch: "
                f"source={len(src_param_ids)}, target={len(dst_param_ids)}"
            )

        src_beta = src_group.get("momentum", None)
        dst_momentum = dst_group.get("momentum", 0.0)
        dst_dampening = dst_group.get("dampening", 0.0)

        if src_beta is None:
            print(
                "[WARN] Source param group has no 'momentum' entry. "
                "Using target SGD momentum for Muon EMA conversion."
            )
            src_beta = dst_momentum

        if abs(float(src_beta) - float(dst_momentum)) > 1e-12:
            print(
                f"[WARN] Source Muon momentum ({src_beta}) differs from target "
                f"SGD momentum ({dst_momentum}). A single buffer rescale is only "
                "exact when these match."
            )

        if dst_dampening != 0:
            print(
                f"[WARN] Target SGD dampening is {dst_dampening}. "
                "Muon EMA -> SGD buffer conversion is exact for dampening=0; "
                "using the steady-state dampening scale."
            )

        if convert_ema_to_sgd_buffer:
            if float(src_beta) >= 1.0:
                raise ValueError(
                    f"Cannot convert Muon EMA buffer with momentum={src_beta}; "
                    "expected momentum < 1."
                )
            sgd_buffer_scale = (1.0 - float(dst_dampening)) / (1.0 - float(src_beta))
        else:
            sgd_buffer_scale = 1.0

        for src_pid, dst_pid, dst_param in zip(src_param_ids, dst_param_ids, live_params):
            src_param_state = src_state.get(src_pid, None)

            if src_param_state is None:
                skipped += 1
                continue

            momentum = src_param_state.get("momentum_buffer", None)

            if momentum is None:
                skipped += 1
                continue

            if not torch.is_tensor(momentum):
                skipped += 1
                continue

            if tuple(momentum.shape) != tuple(dst_param.shape):
                print(
                    f"[WARN] Shape mismatch for param id {src_pid}->{dst_pid}: "
                    f"source momentum shape={tuple(momentum.shape)}, "
                    f"target param shape={tuple(dst_param.shape)}. Skipping."
                )
                skipped += 1
                continue

            momentum = momentum.detach().to(device=dst_param.device, dtype=dst_param.dtype)
            momentum = momentum * scale * sgd_buffer_scale

            # SGD only needs momentum_buffer.
            new_state[dst_pid] = {
                "momentum_buffer": momentum.clone()
            }

            transferred += 1

    dst_sd["state"] = new_state

    # This maps integer state_dict param IDs back onto the live SGD parameters.
    sgd_optimizer.load_state_dict(dst_sd)

    print(
        f"[INFO] Muon -> SGD momentum transfer complete. "
        f"Transferred {transferred} tensors, skipped {skipped}. "
        f"source_itr={ckpt['itr']}, scale={scale}, "
        f"convert_ema_to_sgd_buffer={convert_ema_to_sgd_buffer}"
    )

    return {
        "transferred": transferred,
        "skipped": skipped,
        "source_itr": ckpt["itr"],
    }
