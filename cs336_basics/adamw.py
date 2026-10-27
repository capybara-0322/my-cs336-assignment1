from collections.abc import Callable
from typing import Any, Callable

import torch
from torch import nn
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, weight_decay = 0.01, betas: tuple[float,float] = (0.9, 0.999), eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"alpha":lr, "beta1":betas[0], "beta2":betas[1], "epsilon":eps, "lamb":weight_decay}
        super().__init__(params, defaults)


    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]
            lamb = group["lamb"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) + 1
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                at = alpha * math.sqrt(1-beta2**t)/(1-beta1**t)
                with torch.no_grad():
                    p.mul_(1 - alpha * lamb)
                grad = p.grad.data
                m = beta1 * m + (1-beta1)*grad
                v = beta2 * v + (1-beta2)*grad*grad
                state["m"] = m
                state["v"] = v
                state["t"] = t
                with torch.no_grad():
                    p.sub_(at*m/(torch.sqrt(v)+epsilon))

        return loss












