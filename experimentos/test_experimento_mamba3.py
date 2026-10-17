"""Verificações do protocolo da rede Mamba-3: alinhamento, origens, causalidade e continuidade do estado."""
import unittest

import numpy as np
import torch

from experimento_linear import Persistence, evaluate, evaluation_origins
from experimento_mamba3 import Mamba3Regressor, Normalizer, rollout_targets, simulate, training_windows

SMALL = {"d_model": 16, "n_layers": 2, "d_state": 16, "expand": 2, "headdim": 16}
SETTINGS = {"warmup": 64, "origin_stride": 16, "horizons": [1, 5, 10]}
HISTORY = 32


class Mamba3ProtocolChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.u = rng.choice([-1.0, 1.0], size=(2, 200))
        self.y = rng.normal(0, 0.3, size=(2, 201))
        self.norm = Normalizer({"u": {"mean": 0.1, "std": 0.9}, "y": {"mean": -0.05, "std": 0.3}})
        torch.manual_seed(3)
        self.model = Mamba3Regressor(**SMALL)

    def test_training_window_pairs_u_and_y_at_t_with_target_at_t_plus_1(self):
        x, target = training_windows(self.u, self.y, self.norm, window=64, stride=32)
        self.assertEqual(x.shape, (2 * 5, 64, 2))
        # Segunda janela da segunda trajetória começa em t=32.
        np.testing.assert_allclose(x[6, :, 0], self.norm.u(self.u[1, 32:96]))
        np.testing.assert_allclose(x[6, :, 1], self.norm.y(self.y[1, 32:96]))
        np.testing.assert_allclose(target[6], self.norm.y(self.y[1, 33:97]))

    def test_origins_and_targets_match_the_reference_evaluation(self):
        origins = evaluation_origins(self.u.shape[1], SETTINGS)
        reference = evaluate(Persistence(), self.u, self.y, SETTINGS)["target"]
        np.testing.assert_array_equal(rollout_targets(self.y, origins, max(SETTINGS["horizons"])), reference)

    def test_simulation_reads_no_measurement_after_the_origin(self):
        origin = 100
        hidden = self.y.copy()
        hidden[:, origin + 1:] = np.nan
        normal = simulate(self.model, self.u, self.y, [origin], HISTORY, 10, self.norm, "cpu")
        blind = simulate(self.model, self.u, hidden, [origin], HISTORY, 10, self.norm, "cpu")
        self.assertTrue(np.isfinite(blind).all())
        np.testing.assert_array_equal(blind, normal)

    def test_step_continuation_matches_recomputing_the_whole_prefix(self):
        # Referência lenta: a cada passo, recalcula a sequência inteira com a previsão anexada.
        origin, horizon = 100, 8
        fast = simulate(self.model, self.u[:1], self.y[:1], [origin], HISTORY, horizon, self.norm, "cpu")[0]
        sequence = self.norm.pairs(self.u[0, origin - HISTORY + 1:origin + 1], self.y[0, origin - HISTORY + 1:origin + 1])
        slow = []
        with torch.no_grad():
            for j in range(horizon):
                y_hat = self.model(torch.tensor(sequence[None], dtype=torch.float32))[0, -1].item()
                slow.append(self.norm.y_physical(y_hat))
                sequence = np.vstack([sequence, [self.norm.u(self.u[0, origin + j + 1]), y_hat]])
        np.testing.assert_allclose(fast, slow, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
