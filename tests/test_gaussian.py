import numpy as np
import torch
from cre_reg.experiments import two_gaussian_mixture as tgm


def test_calculate_num_samples():
    N = 103
    alphas = np.random.rand(10)
    alphas /= np.sum(alphas)

    num_samples = tgm.calculate_num_samples(alphas, N)
    # Check that we have the correct number of samples
    assert np.abs(np.sum(num_samples) - N) < 1e-8

    # Check that we are less than 1 away from the ``true'' value
    difference_from_continuous = np.abs(num_samples - alphas * N)
    assert np.all(difference_from_continuous < 1)


class TestNGaussian:
    def test_variance(self):
        alphas = np.random.rand(3)
        mus = np.zeros(3)
        sigmas = np.ones(3)
        ngm = tgm.NGaussianModel(3, alphas, mus, sigmas)

        samples = ngm.sample(10000)
        empirical_std = torch.std(samples)
        empirical_mean = torch.mean(samples)

        assert torch.abs(empirical_mean) < 1e-1
        assert torch.abs(empirical_std - 1) < 1e-1
