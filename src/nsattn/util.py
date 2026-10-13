import math

import torch
import matplotlib.pyplot as plt

def mask_to_img(mask: torch.Tensor, path: str, title: str):
    """Convert a boolean attention mask to a matplotlib image and save it.
    
    Args:
        mask: Boolean tensor of shape [..., L, S] representing an attention mask
        path: Path where to save the image file
    """
    img = mask.cpu().numpy().astype(float)
    h, w = img.shape[-2:]
    aspect = w / h
    aspect **= 0.5
    plt.figure(figsize=(12, 12/aspect))
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position') 
    plt.savefig(path)
    plt.close()


def describe(t: torch.Tensor, name: str = "", /) -> str:
    """Return a compact human-readable description of a tensor's shape, dtype,
    device, value range, mean, std, memory footprint, and gradient status.

    Example output:
        `x` tensor[2, 128, 1536] n=393216 (1.5Mb) x∈[-4.493, 5.403] μ=0.001 σ=1.000 cuda:0

    Args:
        t: The tensor to describe.
        name: Optional short label prepended to the description (e.g. "x").

    Returns:
        Formatted summary string.
    """
    n = t.numel()
    elem_bytes = t.element_size()
    total_bytes = n * elem_bytes
    if total_bytes >= 2**30:
        mem = f"{total_bytes / 2**30:.1f}Gb"
    elif total_bytes >= 2**20:
        mem = f"{total_bytes / 2**20:.1f}Mb"
    elif total_bytes >= 2**10:
        mem = f"{total_bytes / 2**10:.1f}Kb"
    else:
        mem = f"{total_bytes}B"
    parts = [f"tensor{list(t.shape)}", f"n={n} ({mem})"]
    if t.is_floating_point():
        parts.append(f"x∈[{t.min().item():.3g}, {t.max().item():.3g}]")
        parts.append(f"μ={t.mean().item():.3g}")
        if t.numel() > 1:
            parts.append(f"σ={t.std().item():.3g}")
        else:
            parts.append(f"σ=—")
    parts.append(str(t.device))
    if t.requires_grad:
        grad_info = "grad"
        if t.grad is not None:
            grad_info += f" {t.grad.shape}"
        parts.append(grad_info)
    label = f"`{name}` " if name else ""
    return label + " ".join(parts)
