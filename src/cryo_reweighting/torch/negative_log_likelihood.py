import torch
from typing import Optional
from cryo_reweighting.torch.utils import normalize_weights


def evaluate_nll(
    log_weights: torch.Tensor,
    log_Pij: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the negative log-likelihood of the data given the weights.

    Parameters
    ----------
    log_weights: torch.Tensor
        Log of the weights of the clusters.
    log_Pij: torch.Tensor
        Log-likelihood of generating image i from cluster j.
    cluster_size: torch.Tensor
        Number of images in each cluster
    anchor_strength: float
        Strength of the anchor term anchoring the average log weight to zero.
        (Prevents them going to infinity)

    Returns
    -------
    neg_total_ll: torch.Tensor

    """
    # Normalize the log weights
    weighted_alphas = normalize_weights(log_weights)
    log_weighted_alphas = torch.log(weighted_alphas)

    # Evaluate the log-likelihood
    likelihood_per_image = torch.logsumexp(log_Pij + log_weighted_alphas, axis=1)
    neg_total_ll = -1 * torch.mean(likelihood_per_image)
    return neg_total_ll


def eval_naive_log_Pij(
    sq_dist_matrix: torch.Tensor, noise_stdev: float
) -> torch.Tensor:
    """
    Evaluate the likelihood of generating image i from cluster j.

    Parameters
    ----------
    sq_dist_matrix: torch.Tensor
        Matrix of squared distances between cluster and image centers.
        We follow the convention that the first dimension is the image
        and the second dimension is the cluster, which is the opposite
        of the cryoER convention.
    noise_stdev: float
        Standard deviation of the noise in the images.

    Returns
    -------
    log_Pij_matrix: torch.Tensor
        Likelihood of generating image i from cluster j.
    """
    return sq_dist_matrix / (-2 * noise_stdev**2)
