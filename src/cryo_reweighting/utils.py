import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def normalize_weights(log_weights: ArrayLike) -> Array:
    """ Converts the log of weights to weight vector with all positive entries and sums to 1.
    """
    log_weights = jnp.asarray(log_weights) 
    weighted_alphas = jnp.exp(log_weights)
    weighted_alphas = weighted_alphas / jnp.sum(weighted_alphas)
    return weighted_alphas
