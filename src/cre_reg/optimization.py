import torch
from cre_reg.negative_log_likelihood import evaluate_nll
from cre_reg.utils import normalize_weights

from typing import Callable, Optional
import torch.optim as optim
from tqdm import tqdm


def optimize_weights(
    log_Pij: torch.Tensor,
    loss_fxn: Optional[Callable] = None,
    log_weights_init: Optional[torch.Tensor] = None,
    cluster_sizes: Optional[torch.Tensor] = None,
    regularization_fxn: Optional[Callable] = None,
    num_epochs: Optional[int] = 1000,
):
    # Initialize
    if log_weights_init is None:
        log_weights = torch.randn_like(log_Pij[0]) * 0.01
    else:
        log_weights = torch.clone(log_weights_init)

    log_weights.requires_grad = True

    if loss_fxn is None:
        loss_fxn = evaluate_nll

    if regularization_fxn is None:
        regularization_fxn = lambda x: 0

    if cluster_sizes is None:
        cluster_sizes = torch.ones_like(log_weights)
    log_cluster_sizes = torch.log(cluster_sizes)

    # Mak

    # Optimization Loop
    optimizer = optim.Adam([log_weights], lr=0.1)

    losses = []

    for i in tqdm(range(num_epochs)):
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
     This function updates the weights according to the expectation maximization
     algorithm for mixture models.
     For $N$ images and $M$ structures, this updates a given weight m according to
     .. math::

         \alpha_m^{(\text{new})} = \frac{1}{N}\sum_{i=1}^N \frac{\alpha_m p(y_i|x_m)}{\sum_{m'}\alpha_{m'} p(y_i|x_{m'})}
    This is implemented with logarithms of the above equation, for stability.
    """
    weights = normalize_weights(-1 * log_weights_init)
    log_weights = torch.log(weights)

    loss = []
    for k in range(num_iterations):
        log_likelihood_per_image = torch.logsumexp(log_Pij + log_weights, axis=1)
        log_weighted_likelihoods = log_Pij + log_weights
        log_posteriors = log_weighted_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        log_weights = torch.logsumexp(log_posteriors, axis=0)

        # Normalize weights to sum to 1
        weights = normalize_weights(-1 * log_weights)
        log_weights = torch.log(weights)

        # Compute loss
        loss.append(
            torch.sum(torch.exp(log_posteriors) * (log_weights + log_Pij), dim=None)
        )

    loss = torch.tensor(loss)
    return log_weights, loss
