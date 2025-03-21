import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

def normalize_weights(log_weights: ArrayLike) -> Array:
    """
    This is a softmax for when inputs weights are in log form.
    Likely can be replaced with a jax.nn.softmax or similarly!
    """
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights = weights / jnp.sum(weights)
    return weights

