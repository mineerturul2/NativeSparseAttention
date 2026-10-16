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


def sparsity(mask: torch.Tensor) -> float:
    """Return the fraction of masked-out (``False``) entries in a boolean mask.

    This is the mask analogue of a tensor's density: ``1.0`` means fully
    masked out, ``0.0`` means fully attended. It complements ``mask_to_img``
    for quickly reporting how sparse an attention or block mask is.

    Args:
        mask: Boolean tensor whose entries are ``True`` where attention is kept.
    """
    return 1.0 - mask.float().mean().item()


def counts(mask: torch.Tensor) -> tuple[int, int]:
    """Return absolute counts of kept and masked-out entries in a boolean mask.

    Where ``sparsity`` reports the *fraction* of masked-out entries, this
    returns the raw ``(kept, masked)`` counts. Together they make it easy to
    report both the proportion and the absolute number of attention positions
    being ignored for a given mask size.

    Args:
        mask: Boolean tensor whose entries are ``True`` where attention is kept.

    Returns:
        A tuple ``(kept, masked)`` where ``kept`` is the number of ``True``
        entries and ``masked`` is the number of ``False`` entries.
    """
    kept = int(mask.sum().item())
    return kept, int(mask.numel()) - kept


def model_summary(model: torch.nn.Module, /, *, verbose: bool = True) -> str:
    """Return a compact parameter-count summary of a ``torch.nn.Module``.

    When *verbose* is ``True`` (the default), the returned string includes one
    line per submodule with its parameter count, plus a final line for the full
    model total.  When *verbose* is ``False`` only the total line is returned.

    Args:
        model: The PyTorch module to summarise.
        verbose: If ``True``, include a per-child breakdown.

    Returns:
        A multi-line summary string.
    """
    lines: list[str] = []
    total = 0
    trainable = 0

    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.1f}K"
        return str(n)

    for name_, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        t = sum(p.numel() for p in child.parameters() if p.requires_grad)
        total += n
        trainable += t
        if verbose:
            lines.append(f"  {name_:20s}  {_fmt(n):>8s}  params  ({_fmt(t)} trainable)")

    lines.append(f"  {'—' * 30:—>30s}")
    lines.append(f"  {'total':20s}  {_fmt(total):>8s}  params  ({_fmt(trainable)} trainable)")
    return "\n".join(lines)


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
