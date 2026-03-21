import numpy as np
import torch


def heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).float()


def sawtooth(x: torch.Tensor, period: float = 2 * np.pi) -> torch.Tensor:
    return 2 * (x / period - torch.floor(x / period + 0.5))


def weierstrass(
    x: torch.Tensor,
    a: float = 0.5,
    b: int = 7,
    n_terms: int = 20,
) -> torch.Tensor:
    x_64 = x.double()
    result = torch.zeros_like(x_64)
    for n in range(n_terms):
        result = result + a**n * torch.cos(b**n * np.pi * x_64)
    return result.float()


def sin_cos_2d(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.sin(x[:, 0]) * torch.cos(x[:, 1])
    ).unsqueeze(-1)
