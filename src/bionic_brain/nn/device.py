"""Device and NumPy interoperability for the differentiable structures."""
from __future__ import annotations

import torch
from torch import Tensor


def best_device(explicit: str | None = None) -> str:
    """Return an explicit device, else CUDA when available, else CPU."""

    if explicit:
        device = str(explicit)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device not in {"cpu", "cuda", "mps"} and not device.startswith("cuda:"):
            raise ValueError(f"unknown device: {device}")
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def from_numpy(values, *, device: str | None = None, requires_grad: bool = False) -> Tensor:
    """Copy NumPy state into a tensor; spike arrays stay owned by the brain."""

    tensor = torch.as_tensor(values, dtype=torch.float32)
    tensor = tensor.to(device=best_device(device))
    tensor.requires_grad_(requires_grad)
    return tensor


def to_numpy(tensor: Tensor):
    """Detach a learnable result so NumPy simulation state can consume it."""

    return tensor.detach().cpu().numpy()
