import torch
import numpy as np
from cryo_reweighting.torch.negative_log_likelihood import evaluate_nll
from cryo_reweighting.torch.utils import normalize_weights

from typing import Callable, Optional
import torch.optim as optim
from tqdm import tqdm

# TODO: move these two functions into some other folder
# From Robert Gower, slightly modified
def fw_gap(weights, grad):
      """The Frank-Wolfe gap an upper bound on the optimality gap.

      the loss f (negative of the objective) is convex,
          f(y) >= f(x) + <f'(x), y - x>
      We can thus bound the optimality gap by
          f(x) - f(x*) <= - min_y <f'(x), y - x> : y in simplex
      and the RHS is minimized
      """
      # For a convex f, would be
      # -(np.min(grad) - np.inner(grad, param))
      # In our case, grad is the negative of the gradient, so
      # -(np.min(-grad) - np.inner(-grad, param))
      # simplifies to
      # np.max(grad) - np.inner(grad, param)
      # BUT, for ourproblem, np.inner(grad, param) is always 1
      return torch.max(grad) - 1


def grad_log_prob(
    weights: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    Parameters
    ----------
    weights: torch.Tensor
        weights of the clusters.
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from cluster j.

    Returns
    -------
    grad: torch.Tensor

    """
    num_images, num_structures = log_likelihood.shape

    log_weights = torch.log(weights)
    log_density_at_weights = torch.logsumexp(log_likelihood + log_weights, axis=1)

    aux = log_likelihood - log_density_at_weights.reshape(num_images, 1)
    grad =  (1/num_images)*(torch.exp(torch.logsumexp(aux, axis=0)))
    return grad


def multiplicative_gradient(
    log_likelihood,
    tol: Optional[float]=10**-4,
    max_iterations: Optional[int]=20000,
    stats_frequency: Optional[int]=100
)->float:
    
    """
     This function updates the weights according to the expectation maximization
     algorithm for mixture models.
     This is also known as the "multiplicative gradient" method, which has much less notation overload with "EM"!
     
     For $N$ images and $M$ structures, this updates a given weight m according to
     .. math::
         \alpha_m^{(\text{new})} = \frac{1}{N}\sum_{i=1}^N \frac{\alpha_m p(y_i|x_m)}{\sum_{m'}\alpha_{m'} p(y_i|x_{m'})}

    This actually simplifies to, literally multiplying the old guess by the gradient of the log likelihood
     .. math::
         \alpha_m^{(\text{new})} = \alpha_m \nabla L(\alpha)_m,
    where the L(\alpha) is the log likelihood at the old weights

    This is implemented with logarithms of the above equation, for stability.

    By default, the initial weights are set to equal probabilities for all structures, the `most entropic' weights.
 
    Parameters
    ----------
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from cluster j.
    tol: float
        Tolerance for the stopping criteria
    max_iterations: int
        Max iterations if stopping criteria isn't met
    stats_frequency: int:
        Stats are computed at every (stats frequency) iterations
    
    Returns
    -------
    weights: torch.tensor 
    stats_tracking: dictionary
    """
    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*torch.ones(num_structures)
    
    stats_tracking = {}
    stats_tracking["losses"] = []
    stats_tracking["entropies"] = []

    # Iterate
    for k in range(max_iterations):

        # Update weights
        grad = grad_log_prob(weights, log_likelihood)   
        weights = weights*grad

        # Check stopping criterion
        gap = fw_gap(weights,grad)
        if k % stats_frequency == 0: 
            log_weights = torch.log(weights)
            loss = -torch.mean(torch.logsumexp(log_likelihood + log_weights, axis=1))
            entropy = -torch.sum(weights*log_weights)
            stats_tracking["losses"].append(loss)
            stats_tracking["entropies"].append(entropy)
            print(f"#iterations: {k}")
            print(f"loss: {loss}")
            print(f"frank-wolfe gap: {gap}")
            print(f"entropy: {entropy}")
            print("\n")
        
        if gap < tol:
            print("exiting!")
            print(f"#iterations at exit: {k}")
            break

    log_weights = torch.log(weights)
    log_weights = torch.log(normalize_weights(log_weights))
    return log_weights, stats_tracking


# NOTE: will be `depreciated` probably
# NOTE: this function computes the same as multiplicative_gradient, just written differently and with different stats tracked
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

