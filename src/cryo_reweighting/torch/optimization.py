import torch
import numpy as np
from cryo_reweighting.torch.negative_log_likelihood import evaluate_nll
from cryo_reweighting.torch.utils import normalize_weights

from typing import Callable, Optional
import torch.optim as optim
from tqdm import tqdm


def gradient_descent_weights(
    log_Pij: torch.Tensor,
    loss_fxn: Optional[Callable] = None,
    log_weights_init: Optional[torch.Tensor] = None,
    cluster_sizes: Optional[torch.Tensor] = None,
    regularization_fxn: Optional[Callable] = None,
    num_iterations: Optional[int] = 1000,
):
    # Initialize
    if log_weights_init is None:
        log_weights = torch.randn_like(log_Pij[0]) * 0.01
    else:
        log_weights = torch.clone(log_weights_init)

    log_weights.requires_grad = True

    if loss_fxn is None:
        loss_fxn = lambda x: evaluate_nll(x, log_Pij)

    if regularization_fxn is None:
        regularization_fxn = lambda x: 0

    if cluster_sizes is None:
        cluster_sizes = torch.ones_like(log_weights)
    log_cluster_sizes = torch.log(cluster_sizes)

    # Mak

    # Optimization Loop
    optimizer = optim.Adam([log_weights], lr=0.1)

    losses = []

    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()

        log_scaled_weights = log_weights + log_cluster_sizes

        loss = loss_fxn(log_scaled_weights)
        total_loss = loss + regularization_fxn(log_weights)
        total_loss.backward()
        losses.append(total_loss.item())

        optimizer.step()

    return log_weights, torch.tensor(losses)


def expectation_maximization_weights(
    log_Pij: torch.Tensor,
    log_weights_init: Optional[torch.Tensor] = None,
    num_iterations: Optional[int] = 1000,
):
    """
     This function updates the weights according to the expectation maximization algorithm for mixture models.
     For $N$ images and $M$ structures, this updates a given weight m according to
     .. math::

         \alpha_m^{(\text{new})} = \frac{1}{N}\sum_{i=1}^N \frac{\alpha_m p(y_i|x_m)}{\sum_{m'}\alpha_{m'} p(y_i|x_{m'})}
    This is implemented with logarithms of the above equation, for stability.

    The loss function is the negative log likelihood of the weights, 
    .. math::
        
        -1/N*\sum_i \log(\sum_m P_ij \alpha_m)
    By default, the initial weights are set to equal probabilities for all structures, the `most entropic' weights.
    """

    num_images, num_structures = log_Pij.shape

    # Initialize Weights
    if log_weights_init is None:
        log_weights = (1/num_structures)*torch.ones(num_structures)
    else:
        log_weights = torch.clone(log_weights_init)

    # Iterate
    losses = []
    for k in range(num_iterations):
        # Update weights
        log_weighted_likelihoods = log_Pij + log_weights
        log_likelihood_per_image = torch.logsumexp(log_weighted_likelihoods, axis=1)
        log_posteriors = log_weighted_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        log_weights = torch.logsumexp(log_posteriors - np.log(num_images), axis=0)

        # Update loss
        loss = -torch.mean(torch.logsumexp(log_Pij + log_weights, axis=1))
        losses.append(loss.item())

    log_weights = torch.log(normalize_weights(log_weights))
    return log_weights, torch.tensor(losses)


def expectation_maximization_weights_from_clustering(
    log_Pij: torch.Tensor,
    log_weights_init: Optional[torch.Tensor] = None,
    cluster_sizes: Optional[torch.Tensor] = None,
    num_iterations: Optional[int] = 1000,
):
    """
     This function updates the weights according to the expectation maximization
     algorithm for mixture models.
     For $N$ images and $M$ structures, this updates a given weight m according to
     .. math::

         \alpha_m^{(\text{new})} = \frac{1}{N}\sum_{i=1}^N \frac{\alpha_m p(y_i|x_m)}{\sum_{m'}\alpha_{m'} p(y_i|x_{m'})}
    This is implemented with logarithms of the above equation, for stability.

    By default, the initial weights are set to equal probabilities for all structures, the `most entropic' weights.
    NOTE: this function is modified to take in the sizes of clusters
    """

    num_images, num_structures = log_Pij.shape
    if log_weights_init is None:
        log_weights = (1/num_structures)*torch.ones(log_Pij.shape[1])
    else:
        log_weights = torch.clone(log_weights_init)
    if cluster_sizes is None:
        cluster_sizes = torch.ones_like(log_weights)
    log_cluster_sizes = torch.log(cluster_sizes)
    log_scaled_weights = torch.log(normalize_weights(log_weights + log_cluster_sizes))
    losses = []
    for k in range(num_iterations):
        log_weighted_likelihoods = log_Pij + log_scaled_weights
        log_likelihood_per_image = torch.logsumexp(log_weighted_likelihoods, axis=1)
        log_posteriors = log_weighted_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        log_scaled_weights = torch.logsumexp(log_posteriors - np.log(num_images), axis=0)

        loss = -torch.mean(torch.logsumexp(log_Pij + log_scaled_weights, axis=1))
        losses.append(loss.item())

    # Get back weights without cluster sizes
    log_weights = torch.log(normalize_weights(log_scaled_weights - log_cluster_sizes))
    return log_weights, torch.tensor(losses)

