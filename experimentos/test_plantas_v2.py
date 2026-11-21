"""Verificações da planta com saturação e atrito, do multisseno e das configurações do E008 e do E009.

Só NumPy/SciPy. A geração completa só acontece em memória (regra do ruído); tudo o que é gravado vai para
pastas temporárias.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import plantas_v2
from experimento_linear import discretize
from plantas_v2 import (multisine, multisine_values, noise_rule, saturation_friction_plant,
                        simulate_saturation_friction)
from protocolo_v2 import (EXCITATIONS, MANIFEST_FILE, PLANTS, TEST_FILE, aprbs, generate_dataset,
                          load_development, validate_config)

HERE = Path(__file__).resolve().parent
E008 = {"aprbs": "config-saturacao-atrito-aprbs-v2.json", "multisseno": "config-saturacao-atrito-multisseno-v2.json"}
E009 = {0.005: "config-duffing-ruido-0005-v2.json", 0.01: "config-duffing-ruido-001-v2.json",
        0.04: "config-duffing-ruido-004-v2.json"}


def read_config(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def small_config(kind="aprbs", noise=None, seed=11):
    """Configuração do E008 reduzida. O multisseno curto mantém a banda (até cerca de 2 Hz) num período de 512."""
    config = read_config(E008[kind])
    config["seed"] = seed
    steps = 512 if kind == "multisseno" else 400
    config["data"].update(train_episodes=3, validation_episodes=2, test_episodes=2, steps=steps)
    if noise is not None:
        config["data"]["measurement_noise_std"] = noise
    if kind == "multisseno":
        config["excitation"].update(period=512, last_harmonic=51)
    validate_config(config)
    return config


def energy(x, plant):
    return 0.5 * plant["mass"] * x[..., 1] ** 2 + 0.5 * plant["stiffness"] * x[..., 0] ** 2


class PlantChecks(unittest.TestCase):
    def setUp(self):
        self.plant = read_config(E008["aprbs"])["plant"]

    def test_registered_on_import(self):
        self.assertIs(PLANTS["saturation_friction"], saturation_friction_plant)
        self.assertIs(EXCITATIONS["multisine"], multisine)
        self.assertIn("duffing", PLANTS)
        self.assertIn("aprbs", EXCITATIONS)

    def test_matches_exact_linear_discretization_without_saturation_and_friction(self):
        """Fc = 0 e |u| < limite: a planta é a linear do E001, cuja ZOH exata é `discretize`."""
        linear = read_config("config-linear.json")["plant"]
        plant = dict(self.plant, coulomb_friction=0.0, **{k: linear[k] for k in ("mass", "damping", "stiffness",
                                                                                   "sample_time")})
        rng = np.random.default_rng(3)
        u = np.stack([aprbs(rng, 600, {"amplitude": 1.0, "hold_min": 5, "hold_max": 40}) for _ in range(4)])
        self.assertLess(np.abs(u).max(), plant["saturation_limit"])
        x0 = rng.uniform(-0.2, 0.2, (4, 2))
        x = simulate_saturation_friction(u, x0, plant)
        ad, bd = discretize(linear)
        exact = np.empty_like(x)
        exact[:, 0] = x0
        for t in range(u.shape[1]):
            exact[:, t + 1] = exact[:, t] @ ad.T + u[:, t, None] * bd
        np.testing.assert_allclose(x, exact, rtol=0, atol=1e-10)

    def test_substeps_converge(self):
        """20, 40 e 80 subpassos nas trajetórias de treino de uma geração reduzida, nas duas excitações."""
        for kind in E008:
            config = small_config(kind)
            plant = config["plant"]
            self.assertEqual(plant["rk4_substeps"], 20)
            data, states, _ = generate_dataset(saturation_friction_plant(config), EXCITATIONS[config["excitation"]["type"]],
                                               config)
            u, x0 = data["treino_u"], states["treino_x"][:, 0]
            x = {n: simulate_saturation_friction(u, x0, plant, substeps=n) for n in (20, 40, 80)}
            np.testing.assert_array_equal(x[20], states["treino_x"])
            err20 = np.abs(x[20] - x[80])[..., 0].max()
            err40 = np.abs(x[40] - x[80])[..., 0].max()
            with self.subTest(kind=kind):
                self.assertLess(err40, err20)
                self.assertGreater(err20 / err40, 8)  # perto da ordem 4 do RK4 (16)
                # Erro de integração muito menor que o ruído do E008 (pelo menos mil vezes).
                self.assertLess(err20, read_config(E008[kind])["data"]["measurement_noise_std"] / 1000)

    def test_constant_input_above_limit_settles_at_saturated_force(self):
        """Perto de v = 0 o atrito suavizado age como amortecimento viscoso Fc/vε = 30 N·s/m: a posição
        rasteja até o equilíbrio com constante de tempo (c + Fc/vε)/k ≈ 7,6 s. Por isso 300 s."""
        u = np.array([3.0, -5.0, 1.5])[:, None] * np.ones((3, 6000))
        x = simulate_saturation_friction(u, np.array([[0.1, 0.0], [0.0, 0.2], [-0.2, -0.1]]), self.plant)
        expected = np.array([1.5, -1.5, 1.5]) / self.plant["stiffness"]
        np.testing.assert_allclose(x[:, -1, 0], expected, rtol=0, atol=1e-9)
        np.testing.assert_allclose(x[:, -1, 1], 0.0, rtol=0, atol=1e-9)

    def test_friction_dissipates_energy_with_zero_input(self):
        x0 = np.array([[0.3, 0.0], [0.0, 0.5], [-0.2, 0.3]])
        u = np.zeros((3, 400))
        no_damping = dict(self.plant, damping=0.0)
        with_friction = simulate_saturation_friction(u, x0, no_damping)
        without = simulate_saturation_friction(u, x0, dict(no_damping, coulomb_friction=0.0))
        e_friction, e_free = energy(with_friction, no_damping), energy(without, no_damping)
        # Sem amortecimento nem atrito, a energia se conserva (a menos do RK4).
        np.testing.assert_allclose(e_free / e_free[:, :1], 1.0, rtol=0, atol=1e-8)
        # Com atrito: dE/dt = -Fc v tanh(v/vε) <= 0, amostra a amostra.
        self.assertTrue(np.all(np.diff(e_friction, axis=1) <= 1e-12))
        self.assertTrue(np.all(e_friction[:, -1] < 0.05 * e_friction[:, 0]))
        # Com amortecimento e atrito, enquanto |v| >> vε (primeiros 5 s), dissipa mais que só o amortecimento.
        # Depois não vale sempre: com |v| < vε o atrito suavizado age como amortecimento de 30 N·s/m, o
        # movimento fica superamortecido e a posição rasteja (constante de tempo de cerca de 7,6 s).
        both = energy(simulate_saturation_friction(u, x0, self.plant), self.plant)
        viscous = energy(simulate_saturation_friction(u, x0, dict(self.plant, coulomb_friction=0.0)), self.plant)
        self.assertTrue(np.all(np.diff(both, axis=1) <= 1e-12))
        self.assertTrue(np.all(both[:, 1:101] < viscous[:, 1:101]))


class MultisineChecks(unittest.TestCase):
    def setUp(self):
        self.settings = read_config(E008["multisseno"])["excitation"]

    def test_settings_follow_the_pre_registration(self):
        self.assertEqual((self.settings["first_harmonic"], self.settings["last_harmonic"]), (1, 204))
        self.assertEqual(self.settings["period"], 2048)
        self.assertAlmostEqual(self.settings["rms"], 2 / np.sqrt(3), places=15)
        f0 = 1 / (self.settings["period"] * read_config(E008["multisseno"])["plant"]["sample_time"])
        self.assertAlmostEqual(f0, 0.009765625)
        self.assertAlmostEqual(204 * f0, 1.9921875)

    def test_rms_spectrum_and_period(self):
        u = multisine(np.random.default_rng(5), 2048, self.settings)
        self.assertAlmostEqual(float(np.sqrt(np.mean(u ** 2))), 2 / np.sqrt(3), places=12)
        spectrum = np.abs(np.fft.rfft(u))
        inside = np.zeros(spectrum.size, dtype=bool)
        inside[1:205] = True
        np.testing.assert_allclose(spectrum[inside], spectrum[inside].mean(), rtol=1e-9)  # amplitudes iguais
        self.assertLess(spectrum[~inside].max(), 1e-9 * spectrum[inside].min())
        phases = np.random.default_rng(5).uniform(0, 2 * np.pi, 204)
        n = np.arange(2048)
        np.testing.assert_array_equal(multisine_values(phases, n, self.settings), u)
        np.testing.assert_allclose(multisine_values(phases, n + 2048, self.settings), u, rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(multisine_values(phases, n + 1024, self.settings), u))

    def test_reproducible_by_seed_and_draws_only_the_phases(self):
        first, again = np.random.default_rng(9), np.random.default_rng(9)
        u = multisine(first, 2048, self.settings)
        np.testing.assert_array_equal(u, multisine(again, 2048, self.settings))
        self.assertFalse(np.allclose(u, multisine(np.random.default_rng(10), 2048, self.settings)))
        reference = np.random.default_rng(9)
        reference.uniform(0, 2 * np.pi, 204)
        self.assertEqual(first.bit_generator.state, reference.bit_generator.state)

    def test_one_period_per_trajectory(self):
        with self.assertRaises(ValueError):
            multisine(np.random.default_rng(0), 2000, self.settings)
        with self.assertRaises(ValueError):
            multisine(np.random.default_rng(0), 2048, dict(self.settings, last_harmonic=1024))


class DatasetChecks(unittest.TestCase):
    def test_reproducible_and_noise_does_not_change_clean_trajectories(self):
        for kind in E008:
            with self.subTest(kind=kind):
                config = small_config(kind)
                excitation = EXCITATIONS[config["excitation"]["type"]]
                plant = PLANTS[config["plant"]["model"]](config)
                data, states, manifest = generate_dataset(plant, excitation, config)
                again, states_again, manifest_again = generate_dataset(plant, excitation, config)
                for key in data:
                    np.testing.assert_array_equal(data[key], again[key])
                for key in states:
                    np.testing.assert_array_equal(states[key], states_again[key])
                self.assertEqual(manifest, manifest_again)
                quiet_config = copy.deepcopy(config)
                quiet_config["data"]["measurement_noise_std"] = 0.0
                quiet, quiet_states, _ = generate_dataset(plant, excitation, quiet_config)
                for split in ("treino", "validacao", "teste_final"):
                    np.testing.assert_array_equal(data[f"{split}_u"], quiet[f"{split}_u"])
                    np.testing.assert_array_equal(data[f"{split}_y_limpo"], quiet[f"{split}_y_limpo"])
                    np.testing.assert_array_equal(quiet[f"{split}_y"], quiet[f"{split}_y_limpo"])
                    np.testing.assert_array_equal(states[f"{split}_x"], quiet_states[f"{split}_x"])
                    self.assertFalse(np.array_equal(data[f"{split}_y"], data[f"{split}_y_limpo"]))
                other = generate_dataset(plant, excitation, dict(config, seed=config["seed"] + 1))[0]
                self.assertFalse(np.array_equal(data["treino_u"], other["treino_u"]))

    def test_draw_order_per_trajectory_is_state_excitation_noise(self):
        for kind in E008:
            with self.subTest(kind=kind):
                config = small_config(kind)
                excitation = EXCITATIONS[config["excitation"]["type"]]
                data, states, _ = generate_dataset(saturation_friction_plant(config), excitation, config)
                d, limits = config["data"], config["initial_state"]
                children = np.random.SeedSequence(config["seed"]).spawn(7)
                for split, episode in (("treino", 0), ("treino", 2), ("validacao", 1), ("teste_final", 1)):
                    child = children[{"treino": 0, "validacao": 3, "teste_final": 5}[split] + episode]
                    rng = np.random.default_rng(child)
                    x0 = [rng.uniform(-limits["position_limit"], limits["position_limit"]),
                          rng.uniform(-limits["velocity_limit"], limits["velocity_limit"])]
                    u = excitation(rng, d["steps"], config["excitation"])
                    noise = rng.normal(0, d["measurement_noise_std"], d["steps"] + 1)
                    np.testing.assert_array_equal(states[f"{split}_x"][episode, 0], x0)
                    np.testing.assert_array_equal(data[f"{split}_u"][episode], u)
                    np.testing.assert_array_equal(data[f"{split}_y"][episode],
                                                  data[f"{split}_y_limpo"][episode] + noise)


class ConfigChecks(unittest.TestCase):
    def test_new_configs_pass_validation(self):
        for name in list(E008.values()) + list(E009.values()):
            with self.subTest(config=name):
                config = read_config(name)
                validate_config(config)
                self.assertIn(config["plant"]["model"], PLANTS)
                self.assertIn(config["excitation"]["type"], EXCITATIONS)

    def test_e008_follows_the_pre_registration(self):
        reference = read_config("config-duffing-v2.json")
        for kind, seed in (("aprbs", 20261122), ("multisseno", 20261123)):
            with self.subTest(kind=kind):
                config = read_config(E008[kind])
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["plant"], {"model": "saturation_friction", "mass": 1.0, "damping": 0.4,
                                                   "stiffness": 4.0, "saturation_limit": 1.5,
                                                   "coulomb_friction": 0.3, "friction_velocity_scale": 0.01,
                                                   "sample_time": 0.05, "rk4_substeps": 20})
                for key in ("initial_state", "evaluation", "bootstrap", "grids", "pre_registration"):
                    self.assertEqual(config[key], reference[key])
                expected_data = dict(reference["data"], measurement_noise_std=config["data"]["measurement_noise_std"])
                self.assertEqual(config["data"], expected_data)
        self.assertEqual(read_config(E008["aprbs"])["excitation"], reference["excitation"])

    def test_e008_noise_follows_the_rule(self):
        """0,07 × desvio da posição sem ruído do treino, dois algarismos, numa geração sem ruído em memória."""
        for kind, expected in (("aprbs", 0.025), ("multisseno", 0.011)):
            with self.subTest(kind=kind):
                config = read_config(E008[kind])
                rule = noise_rule(config)
                self.assertEqual(rule["trajetorias_treino"], 20)
                self.assertEqual(rule["desvio_ruido_m"], expected)
                self.assertEqual(config["data"]["measurement_noise_std"], rule["desvio_ruido_m"])

    def test_e009_points_only_change_the_noise_and_labels(self):
        reference = read_config("config-duffing-ruido-v2.json")
        for noise, name in E009.items():
            with self.subTest(noise=noise):
                config = read_config(name)
                self.assertEqual(config["data"]["measurement_noise_std"], noise)
                self.assertTrue(config["scenario"].startswith("E009"))
                stripped = {k: v for k, v in config.items() if k not in ("scenario", "description")}
                stripped["data"] = dict(stripped["data"], measurement_noise_std=0.02)
                self.assertEqual(stripped, {k: v for k, v in reference.items()
                                            if k not in ("scenario", "description")})


class CommandChecks(unittest.TestCase):
    def test_generate_command_produces_the_development_layout(self):
        env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        for kind in E008:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                config = small_config(kind)
                config_path = Path(tmp) / "config.json"
                config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                out = Path(tmp) / "dados"
                result = subprocess.run([sys.executable, str(HERE / "plantas_v2.py"), "gerar", "--config",
                                         str(config_path), "--output", str(out)],
                                        capture_output=True, text=True, env=env, cwd=tmp)
                self.assertEqual(result.returncode, 0, result.stderr)
                development, manifest = load_development(out)
                d = config["data"]
                self.assertEqual(sorted(development), sorted(f"{s}_{k}" for s in ("treino", "validacao")
                                                              for k in ("u", "y", "ids")))
                self.assertEqual(development["treino_u"].shape, (d["train_episodes"], d["steps"]))
                self.assertEqual(development["validacao_y"].shape, (d["validation_episodes"], d["steps"] + 1))
                self.assertTrue(manifest["ruido"])
                self.assertEqual(len(manifest["trajetorias"]),
                                 d["train_episodes"] + d["validation_episodes"] + d["test_episodes"])
                self.assertTrue((out / TEST_FILE).is_file())
                self.assertTrue((out / MANIFEST_FILE).is_file())
                self.assertEqual((out / "config.json").read_bytes(), config_path.read_bytes())
                # O mesmo conteúdo da geração em memória.
                data, _, _ = generate_dataset(saturation_friction_plant(config),
                                              EXCITATIONS[config["excitation"]["type"]], config)
                for key in development:
                    np.testing.assert_array_equal(development[key], data[key])
                with self.assertRaises(subprocess.CalledProcessError):  # pasta existente é recusada
                    subprocess.run([sys.executable, str(HERE / "plantas_v2.py"), "gerar", "--config",
                                    str(config_path), "--output", str(out)],
                                   capture_output=True, env=env, cwd=tmp, check=True)


if __name__ == "__main__":
    unittest.main()
