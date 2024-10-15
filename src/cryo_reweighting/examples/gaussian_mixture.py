import jax.numpy as jnp
import numpy as np
import jax
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt

def GaussianMixture1D(weights, means, std_devs):
    """
    TODO
    """
    weights = np.array(weights)
    means = np.array(means)
    std_devs = np.array(std_devs)
    mixing_dist = dist.Categorical(probs=weights/jnp.sum(weights))
    component_dist = dist.Normal(loc=means, scale=std_devs) 
    return dist.MixtureSameFamily(mixing_dist, component_dist) 

def main():

    # try 2 gaussian mixture 
    weights = [0.3, 0.7]
    means = [-1.0, 1.0]
    std_devs = [0.5, 0.5]
    mixture = GaussianMixture1D(weights, means, std_devs)
    samples = mixture.sample(jax.random.PRNGKey(42), (10000,))
 
    # compute pdf output on a 1d grid
    x = jnp.linspace(-4, 4, 100)
    dx = x[1] - x[0]
    
    # normalize output to integrate to 1 on the grid 
    y = np.exp(mixture.log_prob(x))
    y /= dx*y.sum() 
    
    plt.plot(x, y, label='true', color="C0")
    plt.hist(samples, bins=100, range=(-4, 4), density=True, label='hist', color="C1")
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()