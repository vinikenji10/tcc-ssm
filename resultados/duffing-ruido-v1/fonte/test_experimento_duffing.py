"""Verificações do simulador de Duffing e do NARX polinomial, sem depender dos resultados gravados."""
import json
import unittest
from pathlib import Path

import numpy as np

from experimento_duffing import PolynomialNARX, duffing_step, fit_narx, generate_dataset, monomials
from experimento_linear import discretize, rollout


class DuffingChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent / "config-duffing.json").read_text())
        self.config["data"].update(train_episodes=2, validation_episodes=1, test_episodes=1, steps=128)
        self.config["evaluation"].update(warmup=32, horizons=[1, 10, 50])

    def test_without_cubic_term_matches_exact_zoh(self):
        plant = {**self.config["plant"], "cubic_stiffness": 0.0}
        ad, bd = discretize(plant)
        x0 = np.array([0.13, -0.07])
        np.testing.assert_allclose(duffing_step(x0, 0.8, plant), ad @ x0 + bd * 0.8, atol=1e-9)

    def test_static_equilibrium_solves_cubic(self):
        p = self.config["plant"]
        position = 0.3
        force = p["stiffness"] * position + p["cubic_stiffness"] * position ** 3
        np.testing.assert_allclose(duffing_step(np.array([position, 0.0]), force, p), [position, 0.0], atol=1e-12)

    def test_substeps_converge(self):
        x0, force = np.array([0.5, 0.4]), 1.7
        coarse = duffing_step(x0, force, self.config["plant"])
        fine = duffing_step(x0, force, {**self.config["plant"], "rk4_substeps": 200})
        np.testing.assert_allclose(coarse, fine, atol=1e-8)

    def test_dataset_is_reproducible_and_inputs_vary_in_amplitude(self):
        first, _, manifest = generate_dataset(self.config)
        second, _, _ = generate_dataset(self.config)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len({m["id"] for m in manifest}), 4)
        self.assertGreater(len(np.unique(first["treino_u"])), 4)
        self.assertLessEqual(np.abs(first["treino_u"]).max(), self.config["data"]["input_amplitude"])

    def test_monomials_count_includes_constant(self):
        self.assertEqual(len(monomials(4, 3)), 35)  # C(4 + 3, 3)
        self.assertEqual(len(monomials(4, 1)), 5)   # quatro termos lineares e a constante

    def test_narx_recovers_polynomial_map(self):
        rng = np.random.default_rng(0)
        u = rng.uniform(-1, 1, (3, 400))
        y = np.zeros((3, 401))
        for t in range(1, 400):
            y[:, t + 1] = 0.5 * y[:, t] - 0.2 * y[:, t - 1] + 0.3 * u[:, t] - 0.4 * y[:, t] ** 3 + 0.1 * u[:, t - 1] * y[:, t]
        model = fit_narx(u, y, order=2, degree=3)
        prediction = rollout(model, u[0], y[0, :101], 100, 50)
        np.testing.assert_allclose(prediction, y[0, 101:151], atol=1e-10)
        copy = PolynomialNARX.from_dict(model.to_dict())
        self.assertEqual(copy.predict_step(u[0, :50], y[0, :50]), model.predict_step(u[0, :50], y[0, :50]))


if __name__ == "__main__":
    unittest.main()
