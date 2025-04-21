import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import jax
import scipy

from typing import Optional, Union


@jax.jit
def grad_log_prob(weights: ArrayLike,  counts: ArrayLike, likelihood: ArrayLike) -> Array:
    """
    Evaluate the gradient of the log-likelihood of the data given the weights, and counts per image.

    This computes the "probabilistic model" for the image prob density with weights w, counts c
    - (\sum_m p(y_i |x_m) w_m)^c_i
    And then computes the gradient of \sum_i log (\sum_m p(y_i|x_m) w_m)^c_i
    - \sum_i c_i*p(y_i|x_m) / \sum_m p(y_i | x_m) w_m

    Parameters
    ----------
    weights: jax.Array
        weights of the structures.
    counts: jax.Array
        counts of the images.
    likelihood: jax.Array
        (unnormalized)_likelihood of generating image i from cluster j.
        must be of shape (num_images x num_structures) 

    Returns
    -------
    gradient of log marginal likelihood: jax.Array
    """
    model = jnp.sum(likelihood*weights, axis=1)
    grad = jnp.sum((likelihood*counts[:, jnp.newaxis])/model[:, jnp.newaxis], axis=0) 
    return grad


@jax.jit
def update_weights(weights: ArrayLike, grad: ArrayLike) -> Array:
    weights = weights*grad
    return weights


@jax.jit
def update_stats(likelihood: ArrayLike, weights: ArrayLike, counts: ArrayLike):
    #TODO: other stats will go in here
    model = jnp.sum(likelihood*weights, axis=1)
    loss = -jnp.sum(counts*jnp.log(model)) 
    return loss

def multiplicative_gradient(
    log_likelihood: ArrayLike,
    tol: Optional[float] = 1e-8,
    max_iterations: Optional[float] = 100000,
    verbose: Optional[bool]=False,
    counts: Optional[Union[ArrayLike, None]]=None,
    info_freq: Optional[int] = 100
):
    """
    TODO: change docs for counts
    This function updates the weights according to the expectation maximization
    algorithm for mixture models.
    This is also known as the "multiplicative gradient" method, which has much less notation overload with "EM"!
     
    For $N$ images and $M$ structures, this updates a given weight m according to
    .. math::
        w_m^{(\text{new})} = \frac{1}{N}\sum_{i=1}^N \frac{w_m p(y_i|x_m)}{\sum_{m'}w_{m'} p(y_i|x_{m'})}

    This actually simplifies to, literally multiplying the old guess by the gradient of the log likelihood
     .. math::
         w_m^{(\text{new})} = w_m \nabla L(w)_m,
    where the L(w) is the log likelihood at the old weights

    By default, the initial weights are set to equal probabilities for all structures, the `most entropic' weights.
    
    Parameters
    ----------
    log_likelihood: jax.Array
        Log-likelihood of generating image i from cluster j.
    tol: float
        Tolerance for the stopping criteria
    max_iterations: int
        Max iterations if stopping criteria isn't met
    Returns
    -------
    weights: jax.Array 
    """ 

    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*jnp.ones(num_structures)

    # Subtracting the largest entry from each row of likelihood
    # The gradient is invariant to row scaling of likelihood, so this is valid
    # With this, we avoid working in log space for the grad and loss
    log_likelihood = log_likelihood - jnp.max(log_likelihood, 1)[:, jnp.newaxis]
    
    # NOTE: we cannot exponentiate this if previous step hasn't happened 
    likelihood = jnp.exp(log_likelihood)

    if counts is None:
        counts = (1/num_images)*jnp.ones(num_images)

    info = {}
    info["losses"] = []
    info["weights"] = []
    info["weights_idx"] = []
    info["gap"] = []
    
    for k in range(max_iterations):

        # Update weights
        grad = grad_log_prob(weights, counts, likelihood)   
        weights = update_weights(weights, grad)


        # Check stopping criterion: this `gap` is an upper bound on our loss compared to optimal weights
        gap = jnp.max(grad) - 1

        # Update info on optimizatoin
        if k % info_freq == 0:
            loss = update_stats(likelihood, weights)
            print(loss)
            info["losses"].append(loss)
            info["weights"].append(weights)
            info["weights_idx"].append(k)
            info["gap"].append(gap)
            if verbose:
                print(f"Number of iterations:{k}")
                print(f"Gap: {gap}") 
        if gap < tol:
            print(f"exiting at {k} iterations")
            break

    info["losses"] = jnp.stack(info["losses"])
    info["weights"] = jnp.stack(info["weights"])
    info["weights_idx"] = jnp.array(info["weights_idx"])
    info["gap"] = jnp.array(info["gap"])

    return weights, info


@jax.jit
def grad_log_prob_in_log_space(weights,log_likelihood):
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.

    NOTE: this function should output the same as grad_log_prob, but keeping this older version in here just in case

    Parameters
    ----------
    log_weights: jax.Array
        Log of the weights of the clusters.
    log_likelihood: jax.Array
        Log-likelihood of generating image i from cluster j.

    Returns
    -------
    grad: jax.array

    """
    num_images, num_structures = log_likelihood.shape

    log_density_at_weights = jax.scipy.special.logsumexp(a=log_likelihood, b=weights, axis=1)
    aux = log_likelihood - log_density_at_weights.reshape(num_images, 1)
    grad =  (1/num_images)*(jnp.exp(jax.scipy.special.logsumexp(aux, axis=0)))
    return grad

@jax.jit
def update_stats_in_log_space(weights, log_likelihood):
    #TODO: other stats will go in here
    loss = -jnp.mean(jax.scipy.special.logsumexp(a=log_likelihood, b=weights, axis=1))
    return loss

def multiplicative_gradient_in_log_space(
    log_likelihood,
    tol: Optional[float] = 1e-8,
    max_iterations: Optional[float] = 100000,
    verbose: Optional[bool]=False
):
    """
    NOTE: this function should output the same as multiplicative_gradient, but keeping this older version in here just in case   
    """ 

    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*jnp.ones(num_structures)


    info = {}
    info["losses"] = []
    info["weights"] = []
    info["weights_idx"] = []

    for k in range(max_iterations):

        # Update weights
        grad = grad_log_prob_in_log_space(weights, log_likelihood)   
        weights = update_weights(weights, grad)

        # Update info on optimizatoin
        if k % 100 == 0:
            loss = update_stats_in_log_space(weights, log_likelihood)
            info["losses"].append(loss)
            print("NO COUNTS IN LOSS YET")
            info["weights"].append(weights)
            print(loss)
            info["weights_idx"].append(k)

        # Check stopping criterion: this `gap` is an upper bound on our loss compared to optimal weights
        gap = jnp.max(grad) - 1
        if verbose:
            if k % 100 == 0: 
                print(f"Number of iterations:{k}")
                print(f"Gap: {gap}") 
        if gap < tol:
            print(f"Number of iterations: {k}")
            print("exiting!")
            break

    info["weights"] = jnp.stack(info["weights"])
    info["losses"] = jnp.stack(info["losses"])
    info["weights_idx"] = jnp.array(info["weights_idx"])

    return weights, info
