import torch


def gradient_clipping(params, max_norm, eps = 1e-6):
    grads= [p.grad for p in params if p.grad is not None]

    if len(grads) == 0:
        return
    total_norm = torch.sqrt(
        sum(torch.sum(g ** 2) for g in grads)
    )
    clip_coef = max_norm / (total_norm + eps)

    if total_norm > max_norm:
        for g in grads:
            g.mul_(clip_coef)

    
