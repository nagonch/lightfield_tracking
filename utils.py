import torch


def srgb_to_linear(
    srgb: torch.Tensor,
    cutoff: float = 0.04045,
    linear_scale: float = 12.92,
    gamma_offset: float = 0.055,
    gamma_scale: float = 1.055,
    gamma_exponent: float = 2.4,
) -> torch.Tensor:
    below = srgb <= cutoff
    linear = torch.empty_like(srgb)
    linear[below] = srgb[below] / linear_scale
    linear[~below] = ((srgb[~below] + gamma_offset) / gamma_scale) ** gamma_exponent
    return linear


def linear_to_srgb(
    linear: torch.Tensor,
    cutoff: float = 0.0031308,
    linear_scale: float = 12.92,
    gamma_offset: float = 0.055,
    gamma_scale: float = 1.055,
    gamma_exponent: float = 2.4,
) -> torch.Tensor:
    below = linear <= cutoff
    srgb = torch.empty_like(linear)
    srgb[below] = linear[below] * linear_scale
    srgb[~below] = (
        gamma_scale * (linear[~below] ** (1.0 / gamma_exponent)) - gamma_offset
    )
    return srgb
