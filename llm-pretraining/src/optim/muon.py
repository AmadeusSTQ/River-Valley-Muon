import math
import torch
from torch.optim import Optimizer
from typing import Dict, Any, Union, Iterable
from .opt_utils import zeropower_via_newtonschulz5

class Muon(Optimizer):
    """
    Pure Muon Optimizer.
    For 2D+ parameters, applies Momentum and Newton-Schulz orthogonalization (Muon).
    For 1D parameters, applies standard SGD with Momentum.
    """
    def __init__(self, 
                params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], 
                lr: float = 3e-4, 
                momentum: float = 0.95):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1

                buf = state['momentum_buffer']
                # Update momentum (using EMA style commonly paired with Muon)
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                if p.ndim >= 2:
                    # Pure Muon logic for 2D+
                    orig_shape = buf.shape
                    if buf.ndim > 2:
                        buf_2d = buf.view(orig_shape[0], -1)
                    else:
                        buf_2d = buf

                    # Apply Newton-Schulz
                    update_2d = zeropower_via_newtonschulz5(buf_2d, steps=5)
                    
                    # Scale based on matrix size exactly like standard Muon
                    rows, cols = buf_2d.size()
                    scale = max(1.0, math.sqrt(rows / cols)) if rows > cols else max(1.0, math.sqrt(cols / rows))
                    update_2d.mul_(scale)

                    # Reshape back to original dimensions
                    update = update_2d.reshape(orig_shape)
                    # Keep dtype/device aligned with parameter tensor for safe in-place update
                    update = update.to(device=p.device, dtype=p.dtype)
                    
                    p.add_(update, alpha=-lr)
                else:
                    # Pure Muon fallback for 1D: Standard Momentum SGD
                    p.add_(buf, alpha=-lr)

        return loss