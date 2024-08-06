from cre_reg.experiments.two_gaussian_mixture import generate_samples, n_generate_samples
import pytest

@pytest.mark.parametrize("N", [32, 49, 1007])
def test_two_gaussian_number_samples(N):
    sigma_1 = sigma_2 = 2
    
    