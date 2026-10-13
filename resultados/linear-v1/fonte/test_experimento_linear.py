"""Verificações físicas e de protocolo, sem depender dos resultados publicados."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from experimento_linear import (ARX, discretize, fit_arx, generate_dataset,
                                normalization_from_training, rollout)


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent / "config-linear.json").read_text())
        self.config["data"].update(train_episodes=2, validation_episodes=1, test_episodes=1, steps=128)
        self.config["evaluation"].update(warmup=32, horizons=[1, 10, 50])

    def test_zoh_matches_independent_rk4_integration(self):
        p = self.config["plant"]
        ad, bd = discretize(p)
        x0 = np.array([0.13, -0.07])
        force = 0.8
        x = x0.copy()
        dt = p["sample_time"] / 100

        def derivative(value):
            return np.array([value[1], (force - p["damping"] * value[1] - p["stiffness"] * value[0]) / p["mass"]])

        for _ in range(100):
            k1 = derivative(x)
            k2 = derivative(x + dt * k1 / 2)
            k3 = derivative(x + dt * k2 / 2)
            k4 = derivative(x + dt * k3)
            x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        np.testing.assert_allclose(ad @ x0 + bd * force, x, atol=2e-12, rtol=2e-12)

    def test_static_equilibrium_is_force_over_stiffness(self):
        ad, bd = discretize(self.config["plant"])
        state = np.array([1 / self.config["plant"]["stiffness"], 0.0])
        np.testing.assert_allclose(ad @ state + bd, state, atol=1e-14)

    def test_arx_identifies_and_simulates_unseen_linear_trajectory(self):
        signals, _, _, _, _ = generate_dataset(self.config)
        model = fit_arx(signals["treino_u"], signals["treino_y"], 2, 2)
        origin = 32
        predicted = rollout(model, signals["teste_u"][0], signals["teste_y"][0, :origin + 1], origin, 50)
        np.testing.assert_allclose(predicted, signals["teste_y"][0, origin + 1:origin + 51], atol=1e-11, rtol=1e-11)

    def test_rollout_feeds_predictions_back(self):
        class IncrementModel:
            def predict_step(self, u_history, y_history):
                return y_history[-1] + 1

        predicted = rollout(IncrementModel(), np.zeros(20), np.array([10.0, 20.0, 30.0]), 2, 4)
        np.testing.assert_array_equal(predicted, [31.0, 32.0, 33.0, 34.0])

    def test_future_measured_outputs_are_rejected(self):
        model = ARX(2, 2, [1.0, 0.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "sem medições futuras"):
            rollout(model, np.zeros(20), np.ones(20), 2, 4)

    def test_future_input_cannot_change_first_prediction(self):
        signals, _, _, _, _ = generate_dataset(self.config)
        model = fit_arx(signals["treino_u"], signals["treino_y"], 2, 2)
        origin = 32
        u = signals["teste_u"][0].copy()
        history = signals["teste_y"][0, :origin + 1]
        a = rollout(model, u, history, origin, 10)
        u[origin + 1:] = 20
        b = rollout(model, u, history, origin, 10)
        self.assertEqual(a[0], b[0])
        self.assertGreater(abs(a[-1] - b[-1]), 0.01)

    def test_normalization_does_not_use_validation_or_test(self):
        signals, _, _, _, _ = generate_dataset(self.config)
        expected = normalization_from_training(signals)
        for split in ("validacao", "teste"):
            for name in ("u", "y"):
                signals[f"{split}_{name}"] *= 1e6
        self.assertEqual(normalization_from_training(signals), expected)

    def test_reproducibility_and_independent_split_ids(self):
        a, _, manifest, _, _ = generate_dataset(self.config)
        b, _, _, _, _ = generate_dataset(copy.deepcopy(self.config))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        ids = [row["id"] for row in manifest]
        seeds = [tuple(row["spawn_key"]) for row in manifest]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(seeds), len(set(seeds)))


if __name__ == "__main__":
    unittest.main()
