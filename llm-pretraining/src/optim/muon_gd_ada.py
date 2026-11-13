import math
from typing import Any, Dict, Iterable, Optional, Union

import torch
from torch.optim import Optimizer

from .opt_utils import zeropower_via_newtonschulz5


class MuonGDAda(Optimizer):
    """
    Per-matrix adaptive Muon/GD optimizer.

    Design choices:
    - Each matrix-like parameter (ndim >= 2) keeps its own mode in {"muon", "gd"}.
    - Muon mode uses the score s(G) = r ||G||_F^2 / ||G||_*^2 as a gate before
      running a trial comparison.
    - GD mode compares trial decreases first, then uses the score to recalibrate psi.
    - Trial losses are evaluated parameter-wise with the provided closure on the
      current batch, which makes the method expensive but suitable for experiments.
    - One-dimensional parameters always use the GD-style momentum update.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-3,
        momentum: float = 0.95,
        gd_lr: Optional[float] = None,
        gd_momentum: float = 0.0,
        weight_decay: float = 0.0,
        psi: float = 1.0,
        adapt_interval: int = 100,
        warmup_steps: int = 0,
        psi_up: float = 2.0,
        psi_down: float = 0.5,
        initial_mode: str = "muon",
        ns_steps: int = 5,
        rank_tol: float = 1e-7,
        compare_on_eval_steps_only: bool = True,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gd_lr is not None and gd_lr <= 0.0:
            raise ValueError(f"Invalid gd learning rate: {gd_lr}")
        if psi < 0.0:
            raise ValueError(f"Invalid psi: {psi}")
        if adapt_interval <= 0:
            raise ValueError(f"adapt_interval must be positive, got {adapt_interval}")
        if psi_up <= 1.0:
            raise ValueError(f"psi_up must be > 1, got {psi_up}")
        if not (0.0 < psi_down < 1.0):
            raise ValueError(f"psi_down must be in (0,1), got {psi_down}")
        if initial_mode not in {"muon", "gd"}:
            raise ValueError(f"initial_mode must be 'muon' or 'gd', got {initial_mode}")
        if ns_steps <= 0:
            raise ValueError(f"ns_steps must be positive, got {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            gd_lr=gd_lr,
            gd_momentum=gd_momentum,
            weight_decay=weight_decay,
            psi=psi,
            adapt_interval=adapt_interval,
            warmup_steps=warmup_steps,
            psi_up=psi_up,
            psi_down=psi_down,
            initial_mode=initial_mode,
            ns_steps=ns_steps,
            rank_tol=rank_tol,
            compare_on_eval_steps_only=compare_on_eval_steps_only,
        )
        super().__init__(params, defaults)

        self.last_loss: Optional[float] = None
        self.last_avg_score: float = 0.0
        self.last_compare_count: int = 0
        self.last_switch_to_gd: int = 0
        self.last_switch_to_muon: int = 0
        self.last_num_muon_params: int = 0
        self.last_num_gd_params: int = 0
        self.last_trial_muon_loss: Optional[float] = None
        self.last_trial_gd_loss: Optional[float] = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                current_loss = closure()
            loss = current_loss.detach()
            self.last_loss = float(loss.item())
        else:
            self.last_loss = None

        decisions = []
        compare_count = 0
        switch_to_gd = 0
        switch_to_muon = 0
        total_score = 0.0
        score_count = 0
        trial_muon_losses = []
        trial_gd_losses = []

        for group in self.param_groups:
            lr = group["lr"]
            gd_lr = lr if group["gd_lr"] is None else group["gd_lr"]
            momentum = group["momentum"]
            gd_momentum = group["gd_momentum"]
            weight_decay = group["weight_decay"]
            psi_default = group["psi"]
            adapt_interval = group["adapt_interval"]
            warmup_steps = group["warmup_steps"]
            psi_up = group["psi_up"]
            psi_down = group["psi_down"]
            initial_mode = group["initial_mode"]
            ns_steps = group["ns_steps"]
            rank_tol = group["rank_tol"]
            compare_on_eval_steps_only = group["compare_on_eval_steps_only"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["mode"] = initial_mode if p.ndim >= 2 else "gd"
                    state["psi"] = float(psi_default)
                    state["muon_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["gd_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                step_id = int(state["step"])

                grad = p.grad.detach()
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                muon_buf = state["muon_buffer"]
                gd_buf = state["gd_buffer"]
                muon_buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                gd_buf.mul_(gd_momentum).add_(grad, alpha=1.0)

                muon_update = self._build_muon_update(muon_buf, p, ns_steps)
                gd_update = gd_buf.to(device=p.device, dtype=p.dtype)

                if p.ndim < 2:
                    decisions.append((p, gd_update, gd_lr, "gd", state["psi"]))
                    continue

                score = self._spectral_score(muon_buf, rank_tol)
                total_score += score
                score_count += 1

                mode = state["mode"]
                psi_value = float(state["psi"])
                in_warmup = step_id <= warmup_steps
                is_eval_step = (step_id % adapt_interval == 0)
                can_compare = (
                    closure is not None
                    and is_eval_step
                    and (not compare_on_eval_steps_only or is_eval_step)
                )

                chosen_mode = mode
                chosen_update = muon_update if mode == "muon" else gd_update
                chosen_lr = lr if mode == "muon" else gd_lr
                new_psi = psi_value

                if not in_warmup and can_compare:
                    if mode == "muon":
                        if score >= psi_value:
                            muon_loss = self._trial_loss(closure, p, muon_update, lr)
                            gd_loss = self._trial_loss(closure, p, gd_update, gd_lr)
                            compare_count += 1
                            trial_muon_losses.append(muon_loss)
                            trial_gd_losses.append(gd_loss)
                            if gd_loss < muon_loss:
                                chosen_mode = "gd"
                                chosen_update = gd_update
                                chosen_lr = gd_lr
                                switch_to_gd += 1
                            else:
                                chosen_mode = "muon"
                                chosen_update = muon_update
                                chosen_lr = lr
                                new_psi = psi_value * psi_up
                    else:
                        muon_loss = self._trial_loss(closure, p, muon_update, lr)
                        gd_loss = self._trial_loss(closure, p, gd_update, gd_lr)
                        compare_count += 1
                        trial_muon_losses.append(muon_loss)
                        trial_gd_losses.append(gd_loss)
                        if muon_loss < gd_loss:
                            chosen_mode = "muon"
                            chosen_update = muon_update
                            chosen_lr = lr
                            switch_to_muon += 1
                            if score >= psi_value:
                                new_psi = psi_value * psi_up
                            else:
                                new_psi = psi_value * psi_down
                        else:
                            chosen_mode = "gd"
                            chosen_update = gd_update
                            chosen_lr = gd_lr

                decisions.append((p, chosen_update, chosen_lr, chosen_mode, new_psi))

        for p, update, applied_lr, chosen_mode, new_psi in decisions:
            p.add_(update, alpha=-applied_lr)
            state = self.state[p]
            state["mode"] = chosen_mode
            state["psi"] = max(0.0, float(new_psi))

        self.last_compare_count = compare_count
        self.last_switch_to_gd = switch_to_gd
        self.last_switch_to_muon = switch_to_muon
        self.last_avg_score = total_score / max(1, score_count)
        self.last_num_muon_params = sum(
            1 for group in self.param_groups for p in group["params"] if self.state.get(p, {}).get("mode") == "muon"
        )
        self.last_num_gd_params = sum(
            1 for group in self.param_groups for p in group["params"] if self.state.get(p, {}).get("mode") == "gd"
        )
        self.last_trial_muon_loss = (
            float(sum(trial_muon_losses) / len(trial_muon_losses))
            if trial_muon_losses
            else None
        )
        self.last_trial_gd_loss = (
            float(sum(trial_gd_losses) / len(trial_gd_losses))
            if trial_gd_losses
            else None
        )

        return loss

    def _build_muon_update(self, buf: torch.Tensor, param: torch.Tensor, ns_steps: int) -> torch.Tensor:
        if param.ndim < 2:
            return buf.to(device=param.device, dtype=param.dtype)

        orig_shape = buf.shape
        buf_2d = buf.view(orig_shape[0], -1) if buf.ndim > 2 else buf
        update_2d = zeropower_via_newtonschulz5(buf_2d, steps=ns_steps)
        rows, cols = update_2d.size()
        scale = max(1.0, math.sqrt(rows / cols)) if rows > cols else max(1.0, math.sqrt(cols / rows))
        update_2d = update_2d.mul(scale)
        return update_2d.reshape(orig_shape).to(device=param.device, dtype=param.dtype)

    def _spectral_score(self, grad_like: torch.Tensor, rank_tol: float) -> float:
        grad_2d = grad_like.view(grad_like.shape[0], -1) if grad_like.ndim > 2 else grad_like
        if grad_2d.numel() == 0:
            return 1.0

        svals = torch.linalg.svdvals(grad_2d.float())
        if svals.numel() == 0:
            return 1.0

        sigma_max = float(svals.max().item())
        if sigma_max <= 0.0:
            return 1.0

        threshold = rank_tol * sigma_max
        rank = int(torch.count_nonzero(svals > threshold).item())
        if rank <= 0:
            return 1.0

        fro_sq = float(torch.sum(svals * svals).item())
        nuc = float(torch.sum(svals).item())
        if nuc <= 0.0:
            return 1.0

        return float(rank * fro_sq / (nuc * nuc))

    @torch.no_grad()
    def _trial_loss(
        self,
        closure,
        param: torch.Tensor,
        update: torch.Tensor,
        lr: float,
    ) -> float:
        original = param.detach().clone()
        param.add_(update, alpha=-lr)
        with torch.enable_grad():
            loss = closure()
        param.copy_(original)
        return float(loss.detach().item())
