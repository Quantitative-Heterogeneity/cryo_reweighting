import torch
from typing import Optional


# Try regularizing with entropy.
def normalize_weights(
    negative_log_weights: torch.Tensor,
) -> torch.Tensor:
    """ """
    weighted_alphas = torch.exp(-negative_log_weights)
    weighted_alphas = weighted_alphas / torch.sum(weighted_alphas)
    return weighted_alphas
