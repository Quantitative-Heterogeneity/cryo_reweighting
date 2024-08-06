import numpy as np


def _check_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return


def generate_two_state_matrix():
    eps = 1e-16
    # structure should be twice as likely.
    likelihood = np.array(
        [
            [1, 1e-15],
            [0.5, 1e-15],
            [1e-15, 0.2],
            [0.5, 0.5],
        ]
    )
    answer = np.array([2, 1]) / 3.0

    return np.log(likelihood), np.log(answer)


class Test_Gradient_Descent_Pytorch:
    def test_two_states(self):
        torch = _check_torch()

        from cryo_reweighting.torch.optimization import gradient_descent_weights
        from cryo_reweighting.torch.utils import normalize_weights

        log_Pij, answer = generate_two_state_matrix()
        log_Pij = torch.tensor(log_Pij)
        if torch.cuda.is_available():
            log_Pij = log_Pij.cuda()

        log_weights, losses = gradient_descent_weights(log_Pij, num_iterations=100)
        log_weights = torch.log(normalize_weights(log_weights))
        log_weights = log_weights.cpu().detach().numpy()

        assert np.allclose(log_weights, answer, atol=1e-2)

    def test_two_states_nontrivial_weights(self):
        torch = _check_torch()

        from cryo_reweighting.torch.optimization import gradient_descent_weights
        from cryo_reweighting.torch.utils import normalize_weights

        log_Pij, answer = generate_two_state_matrix()

        log_Pij = torch.tensor(log_Pij)
        if torch.cuda.is_available():
            log_Pij = log_Pij.cuda()

        cluster_sizes = torch.tensor([2.0, 1.0], device=log_Pij.device)

        log_weights, losses = gradient_descent_weights(
            log_Pij, cluster_sizes=cluster_sizes, num_iterations=100
        )
        log_weights = log_weights + torch.log(cluster_sizes)
        log_weights = torch.log(normalize_weights(log_weights))
        print(log_weights)
        log_weights = log_weights.cpu().detach().numpy()

        assert np.allclose(log_weights, answer, atol=1e-2)


class Test_Expecation_Maximization_Pytorch:
    def test_two_states(self):
        torch = _check_torch()

        from cryo_reweighting.torch.optimization import expectation_maximization_weights
        from cryo_reweighting.torch.utils import normalize_weights

        log_Pij, answer = generate_two_state_matrix()
        log_Pij = torch.tensor(log_Pij)
        if torch.cuda.is_available():
            log_Pij = log_Pij.cuda()

        log_weights, losses = expectation_maximization_weights(log_Pij)
        log_weights = torch.log(normalize_weights(log_weights))
        log_weights = log_weights.cpu().detach().numpy()

        assert np.allclose(log_weights, answer, atol=1e-2)

    def test_two_states_nontrivial_weights(self):
        torch = _check_torch()

        from cryo_reweighting.torch.optimization import expectation_maximization_weights
        from cryo_reweighting.torch.utils import normalize_weights

        log_Pij, answer = generate_two_state_matrix()

        log_Pij = torch.tensor(log_Pij)
        if torch.cuda.is_available():
            log_Pij = log_Pij.cuda()

        cluster_sizes = torch.tensor([2.0, 1.0], device=log_Pij.device)

        log_weights, losses = expectation_maximization_weights(
            log_Pij, cluster_sizes=cluster_sizes
        )
        log_weights = log_weights + torch.log(cluster_sizes)
        log_weights = torch.log(normalize_weights(log_weights))
        print(log_weights)
        log_weights = log_weights.cpu().detach().numpy()

        assert np.allclose(log_weights, answer, atol=1e-2)
