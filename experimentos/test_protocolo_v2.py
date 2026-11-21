"""Verificações do protocolo v2, só com NumPy: divisão, reprodutibilidade, contrato, métricas, bootstrap,
divergência e acesso ao teste final. Tudo roda em pastas temporárias, com dados pequenos."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import avaliar_teste_final
import experimento_duffing
from experimento_linear import ARX, Persistence, evaluate, fit_arx
from protocolo_v2 import (ADAPTERS, MANIFEST_FILE, TEST_FILE, USAGE_LOG, StepModelAdapter, aprbs, arrays_sha256,
                          bootstrap_families, duffing_plant, evaluate_members, evaluation_settings, freeze_selection,
                          generate_dataset, h3_verdict, load_development, load_final_test, read_json,
                          register_adapter, rollout_targets, rollout_windows, save_dataset, select_candidate,
                          time_aligned_pairs, validate_config, validation_score, write_json)

HERE = Path(__file__).resolve().parent


def small_config(noise=0.0, seed=7):
    config = json.loads((HERE / "config-duffing-v2.json").read_text(encoding="utf-8"))
    config["seed"] = seed
    config["data"].update(train_episodes=3, validation_episodes=2, test_episodes=4, steps=400,
                          measurement_noise_std=noise)
    config["bootstrap"]["resamples"] = 2000
    validate_config(config)
    return config


def make_dataset(directory, config):
    data, states, manifest = generate_dataset(duffing_plant(config), aprbs, config)
    return save_dataset(Path(directory) / "dados", data, states, manifest, json.dumps(config).encode())


def legacy_config(config):
    """Mesma planta e excitação no formato de `experimento_duffing`."""
    e, d = config["excitation"], config["data"]
    return {"seed": config["seed"], "plant": config["plant"],
            "data": {"train_episodes": d["train_episodes"], "validation_episodes": d["validation_episodes"],
                     "test_episodes": d["test_episodes"], "steps": d["steps"], "input_amplitude": e["amplitude"],
                     "hold_min": e["hold_min"], "hold_max": e["hold_max"],
                     "initial_position_limit": config["initial_state"]["position_limit"],
                     "initial_velocity_limit": config["initial_state"]["velocity_limit"],
                     "measurement_noise_std": d["measurement_noise_std"]},
            "evaluation": {"warmup": 256, "origin_stride": 16, "horizons": [1, 10, 50]},
            "arx": {"output_orders": [2], "input_orders": [2]}, "narx": {"orders": [2], "degrees": [2]}}


def reference_families(arx_name="arx"):
    return {"persistencia": {"adaptador": "persistencia", "escolha": {},
                             "nota_validacao": {"rmse_trajetoria_m": 1.0, "horizonte_passos": 50},
                             "membros": [{"semente": None}]},
            arx_name: {"adaptador": "arx_medido", "escolha": {"output_order": 4, "input_order": 4},
                       "nota_validacao": {"rmse_trajetoria_m": 0.1, "horizonte_passos": 50},
                       "membros": [{"semente": None, "artefatos": {"modelo": "modelo-arx.json"}}]}}


class Probe:
    """Modelo que guarda o que recebeu e recusa entradas não finitas; prevê persistência."""
    name, parameters = "sonda", 0

    def __init__(self):
        self.calls = []

    def simulate(self, y_past, u_past, u_future):
        self.calls.append((y_past.copy(), u_past.copy(), u_future.copy()))
        if not (np.isfinite(y_past).all() and np.isfinite(u_past).all() and np.isfinite(u_future).all()):
            raise AssertionError("Valor não finito chegou ao modelo.")
        return np.repeat(y_past[:, -1:], u_future.shape[1], axis=1)


class Constant:
    def __init__(self, value, name):
        self.value, self.name, self.parameters = value, name, 1

    def simulate(self, y_past, u_past, u_future):
        return np.full(u_future.shape, self.value)


class HashChecks(unittest.TestCase):
    def test_text_hash_ignores_line_endings_and_binary_hash_does_not(self):
        import protocolo_v2
        with tempfile.TemporaryDirectory() as tmp:
            lf, crlf = Path(tmp) / "lf.py", Path(tmp) / "crlf.py"
            lf.write_bytes(b"a = 1\nb = 2\n")
            crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
            self.assertEqual(protocolo_v2.sha256_file(lf), protocolo_v2.sha256_file(crlf))
            self.assertEqual(protocolo_v2.sha256_file(lf), protocolo_v2.sha256_text(crlf.read_bytes()))
            raw_lf, raw_crlf = Path(tmp) / "lf.npz", Path(tmp) / "crlf.npz"
            raw_lf.write_bytes(b"a\nb")
            raw_crlf.write_bytes(b"a\r\nb")
            self.assertNotEqual(protocolo_v2.sha256_file(raw_lf), protocolo_v2.sha256_file(raw_crlf))


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.data, self.states, self.manifest = generate_dataset(duffing_plant(self.config), aprbs, self.config)

    def test_split_by_trajectory_without_intersection(self):
        ids = {s: set(self.data[f"{s}_ids"]) for s in ("treino", "validacao", "teste_final")}
        self.assertEqual(sum(len(v) for v in ids.values()), len(set.union(*ids.values())))
        keys = [tuple(m["spawn_key"]) for m in self.manifest]
        self.assertEqual(len(keys), len(set(keys)))
        rows = np.concatenate([self.data[f"{s}_u"] for s in ("treino", "validacao", "teste_final")])
        self.assertEqual(len(np.unique(rows, axis=0)), len(rows))
        with tempfile.TemporaryDirectory() as tmp:
            out = make_dataset(tmp, self.config)
            with np.load(out / "dados-desenvolvimento.npz") as f:
                self.assertEqual(sorted(f.files), sorted(f"{s}_{k}" for s in ("treino", "validacao")
                                                         for k in ("u", "y", "ids")))
            with np.load(out / TEST_FILE) as f:
                self.assertTrue(all(k.startswith("teste_final_") for k in f.files))
            development, _ = load_development(out)
            self.assertFalse(any("teste" in k for k in development))
            self.assertFalse((out / USAGE_LOG).exists())

    def test_reproducible_by_seed(self):
        again, _, _ = generate_dataset(duffing_plant(self.config), aprbs, self.config)
        for key in self.data:
            np.testing.assert_array_equal(self.data[key], again[key])
        self.assertEqual(arrays_sha256(self.data), arrays_sha256(again))
        other = small_config(seed=8)
        different, _, _ = generate_dataset(duffing_plant(other), aprbs, other)
        self.assertFalse(np.array_equal(self.data["treino_u"], different["treino_u"]))

    def test_plant_function_reproduces_e003_generator(self):
        for noise in (0.0, 0.02):
            config = small_config(noise=noise)
            ours, _, _ = generate_dataset(duffing_plant(config), aprbs, config)
            old, _, _ = experimento_duffing.generate_dataset(legacy_config(config))
            for split, old_split in (("treino", "treino"), ("validacao", "validacao"), ("teste_final", "teste")):
                np.testing.assert_array_equal(ours[f"{split}_u"], old[f"{old_split}_u"])
                np.testing.assert_array_equal(ours[f"{split}_y"], old[f"{old_split}_y"])

    def test_clean_output_stays_out_of_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = make_dataset(tmp, small_config(noise=0.02))
            development, manifest = load_development(out)
            self.assertFalse(any("limpo" in k for k in development))
            self.assertTrue(manifest["ruido"])
            with np.load(out / TEST_FILE) as f:
                self.assertIn("teste_final_y_limpo", f.files)
            self.assertTrue((out / "diagnostico" / "saida-sem-ruido-desenvolvimento.npz").exists())
            with self.assertRaises(FileExistsError):
                make_dataset(tmp, small_config(noise=0.02))


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.settings = evaluation_settings(self.config)
        self.data, _, _ = generate_dataset(duffing_plant(self.config), aprbs, self.config)
        self.arx = fit_arx(self.data["treino_u"], self.data["treino_y"], 4, 4)

    def test_windows_follow_the_time_convention(self):
        episodes, steps, history, horizon = 3, 400, 256, 50
        code = 1000 * np.arange(episodes)[:, None]
        y = (code + np.arange(steps + 1)).astype(float)
        u = -(code + np.arange(steps) + 0.5)
        origins = np.array([256, 300, 350])
        y_past, u_past, u_future = rollout_windows(u, y, origins, history, horizon)
        targets = rollout_targets(y, origins, horizon)
        pairs = time_aligned_pairs(y_past, u_past, u_future)
        for b, (e, t) in enumerate((e, t) for e in range(episodes) for t in origins):
            np.testing.assert_array_equal(y_past[b], 1000 * e + np.arange(t - history + 1, t + 1))
            np.testing.assert_array_equal(u_past[b], -(1000 * e + np.arange(t - history, t) + 0.5))
            np.testing.assert_array_equal(u_future[b], -(1000 * e + np.arange(t, t + horizon) + 0.5))
            np.testing.assert_array_equal(targets[b], 1000 * e + np.arange(t + 1, t + horizon + 1))
            # aquecimento da rede do E002: pares (u[s], y[s]) para s = t-255..t
            np.testing.assert_array_equal(pairs[b, :, 0], u[e, t - history + 1:t + 1])
            np.testing.assert_array_equal(pairs[b, :, 1], y[e, t - history + 1:t + 1])
        with self.assertRaises(ValueError):
            rollout_windows(u, y, [255], history, horizon)

    def test_no_measurement_after_origin_reaches_the_model(self):
        u, y = self.data["validacao_u"], self.data["validacao_y"]
        model = StepModelAdapter(self.arx)
        for t in (256, 304, 336):
            hidden = y.copy()
            hidden[:, t + 1:] = np.nan
            self.assertTrue(np.isnan(rollout_targets(hidden, [t], 50)).all())
            probe = Probe()
            probe.simulate(*rollout_windows(u, hidden, [t], 256, 50))
            original = model.simulate(*rollout_windows(u, y, [t], 256, 50))
            masked = model.simulate(*rollout_windows(u, hidden, [t], 256, 50))
            self.assertTrue(np.isfinite(masked).all())
            np.testing.assert_array_equal(original, masked)

    def test_batch_size_does_not_change_the_result(self):
        u, y = self.data["validacao_u"], self.data["validacao_y"]
        model = StepModelAdapter(self.arx)
        whole = evaluate_members([model], u, y, self.settings)["members"][0]["measured"]["prediction"]
        split = evaluate_members([model], u, y, self.settings, batch_size=5)["members"][0]["measured"]["prediction"]
        np.testing.assert_array_equal(whole, split)

    def test_wrong_output_shape_is_rejected(self):
        class Wrong:
            name, parameters = "errado", 0

            def simulate(self, y_past, u_past, u_future):
                return y_past[:, -10:]
        with self.assertRaises(ValueError):
            evaluate_members([Wrong()], self.data["validacao_u"], self.data["validacao_y"], self.settings)


class MetricChecks(unittest.TestCase):
    def setUp(self):
        self.config = small_config(noise=0.02)
        self.settings = evaluation_settings(self.config)
        self.data, self.states, _ = generate_dataset(duffing_plant(self.config), aprbs, self.config)
        self.arx = fit_arx(self.data["treino_u"], self.data["treino_y"], 4, 4)

    def test_metrics_match_summarize_rollouts_of_e001(self):
        u, y = self.data["validacao_u"], self.data["validacao_y"]
        for model in (Persistence(), self.arx):
            ours = evaluate_members([StepModelAdapter(model)], u, y, self.settings)["members"][0]["measured"]
            old = evaluate(model, u, y, self.settings)
            self.assertEqual(ours["rows"], old["rows"])
            self.assertEqual(ours["rmse_curve"], old["rmse_curve"])
            np.testing.assert_array_equal(ours["prediction"], old["prediction"])

    def test_clean_metrics_match_e004(self):
        u, y, clean = self.data["validacao_u"], self.data["validacao_y"], self.data["validacao_y_limpo"]
        member = evaluate_members([StepModelAdapter(self.arx)], u, y, self.settings, y_clean=clean)["members"][0]
        old = experimento_duffing.clean_metrics(evaluate(self.arx, u, y, self.settings), clean, self.settings)
        self.assertEqual(member["clean"]["rows"], old["rows"])

    def test_squared_errors_reproduce_the_rmse(self):
        u, y = self.data["validacao_u"], self.data["validacao_y"]
        evaluation = evaluate_members([StepModelAdapter(self.arx)], u, y, self.settings)
        member = evaluation["members"][0]
        errors = {"arx": member["squared_errors"][None]}
        for row in member["measured"]["rows"]:
            h = row["horizon_steps"]
            for kind, key in (("trajectory", "rmse_trajectory_m"), ("endpoint", "rmse_endpoint_m")):
                result = bootstrap_families(errors, evaluation["origins_per_episode"], h, kind, resamples=1000)
                self.assertAlmostEqual(result["families"]["arx"]["point"], row[key], places=12)


class DivergenceChecks(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.settings = evaluation_settings(self.config)
        self.data, _, _ = generate_dataset(duffing_plant(self.config), aprbs, self.config)

    def test_validation_discards_and_records_divergent_candidates(self):
        arx = StepModelAdapter(fit_arx(self.data["treino_u"], self.data["treino_y"], 2, 2))
        candidates = [validation_score([Constant(1e9, "explode")], self.data, self.settings),
                      validation_score([Constant(np.nan, "nan")], self.data, self.settings),
                      validation_score([arx], self.data, self.settings)]
        self.assertTrue(candidates[0]["discarded"] and candidates[1]["discarded"])
        self.assertIn("explode", candidates[0]["reason"])
        chosen, record = select_candidate(candidates, self.settings["tie_tolerance_m"])
        self.assertEqual(chosen, 2)
        self.assertEqual([d["index"] for d in record["discarded"]], [0, 1])
        with self.assertRaises(RuntimeError):
            select_candidate(candidates[:2], 1e-12)

    def test_divergence_in_test_is_kept_as_result(self):
        u, y = self.data["teste_final_u"], self.data["teste_final_y"]
        evaluation = evaluate_members([Constant(np.nan, "nan"), StepModelAdapter(Persistence())], u, y,
                                      self.settings, divergence_limit_m=10.0)
        diverged, fine = evaluation["members"]
        self.assertTrue(diverged["divergence"]["diverged"])
        self.assertEqual(len(diverged["measured"]["rows"]), 3)
        self.assertTrue(all(np.isinf(r["rmse_trajectory_m"]) for r in diverged["measured"]["rows"]))
        errors = {"nan": diverged["squared_errors"][None], "persistencia": fine["squared_errors"][None]}
        result = bootstrap_families(errors, evaluation["origins_per_episode"], 50, resamples=1000,
                                    pairs=[("persistencia", "nan")])
        self.assertEqual(result["comparisons"][0]["verdict"], "a_menor")


class BootstrapChecks(unittest.TestCase):
    def test_two_trajectories_have_the_exact_distribution(self):
        # Erros quadráticos 1 e 9: RMSE reamostrado vale 1, √5 ou 3, com probabilidades 1/4, 1/2 e 1/4.
        result = bootstrap_families({"m": np.array([[[1.0], [9.0]]])}, 1, 1, resamples=20000, seed=3)
        values = result["replicates"]["m"]
        for value, probability in ((1.0, 0.25), (np.sqrt(5), 0.5), (3.0, 0.25)):
            self.assertAlmostEqual(np.mean(np.isclose(values, value)), probability, delta=0.015)
        self.assertEqual(result["families"]["m"]["interval"], [1.0, 3.0])
        self.assertAlmostEqual(result["families"]["m"]["point"], np.sqrt(5))

    def test_seeds_only_have_the_exact_distribution(self):
        # Sementes com RMSE 1 e 3 em todas as trajetórias: a média das sementes sorteadas vale 1, 2 ou 3.
        errors = np.stack([np.full((3, 1), 1.0), np.full((3, 1), 9.0)])
        result = bootstrap_families({"rede": errors}, 1, 1, resamples=20000, seed=4)
        values = result["replicates"]["rede"]
        for value, probability in ((1.0, 0.25), (2.0, 0.5), (3.0, 0.25)):
            self.assertAlmostEqual(np.mean(np.isclose(values, value)), probability, delta=0.015)
        family = result["families"]["rede"]
        self.assertAlmostEqual(family["point"], 2.0)
        self.assertAlmostEqual(family["seed_std"], np.sqrt(2))

    def test_variance_of_the_mean_matches_the_ideal_bootstrap(self):
        values = np.random.default_rng(5).uniform(1, 5, 20)
        result = bootstrap_families({"m": values[None, :, None]}, 1, 1, resamples=40000, seed=6)
        mse = result["replicates"]["m"] ** 2
        self.assertAlmostEqual(mse.var() / (values.var() / len(values)), 1.0, delta=0.05)

    def test_paired_difference(self):
        a = np.random.default_rng(7).uniform(0, 1, (1, 10, 3))
        result = bootstrap_families({"a": a, "b": 4 * a, "c": a.copy()}, 2, 3, resamples=5000, seed=8,
                                    pairs=[("a", "b"), ("a", "c")])
        np.testing.assert_allclose(result["replicates"]["b"], 2 * result["replicates"]["a"])
        ab, ac = result["comparisons"]
        self.assertAlmostEqual(ab["difference"], -result["families"]["a"]["point"])
        self.assertEqual(ab["verdict"], "a_menor")
        self.assertEqual(ac["interval"], [0.0, 0.0])
        self.assertEqual(ac["verdict"], "equivalentes")
        again = bootstrap_families({"a": a, "b": 4 * a, "c": a.copy()}, 2, 3, resamples=5000, seed=8,
                                   pairs=[("a", "b"), ("a", "c")])
        self.assertEqual(result["comparisons"], again["comparisons"])

    def test_h3_needs_every_competitor(self):
        comparisons = [{"a": "mamba3", "b": "frols", "verdict": "a_menor"},
                       {"a": "gru", "b": "mamba3", "verdict": "equivalentes"}]
        verdict = h3_verdict(comparisons, "mamba3", ["frols", "gru"])
        self.assertEqual(verdict["verdicts"], {"frols": "a_menor", "gru": "equivalentes"})
        self.assertFalse(verdict["sustained"])
        comparisons[1]["verdict"] = "b_menor"
        self.assertTrue(h3_verdict(comparisons, "mamba3", ["frols", "gru"])["sustained"])


class FinalTestAccessChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = small_config()
        self.data_dir = make_dataset(self.root, self.config)
        development, _ = load_development(self.data_dir)
        arx = fit_arx(development["treino_u"], development["treino_y"], 4, 4)
        write_json(self.root / "modelo-arx.json", arx.to_dict())

    def tearDown(self):
        self.tmp.cleanup()

    def freeze(self, name="selecao-congelada.json", **kwargs):
        return freeze_selection(self.root / name, scenario="teste", data_dir=self.data_dir,
                                families=reference_families(), best_traditional="arx",
                                comparisons=[("arx", "persistencia")], **kwargs)

    def test_requires_frozen_selection(self):
        with self.assertRaises(FileNotFoundError):
            load_final_test(self.data_dir, self.root / "nao-existe.json")
        draft = self.root / "rascunho.json"
        self.freeze()
        selection = read_json(self.root / "selecao-congelada.json")
        write_json(draft, {**selection, "congelada": False})
        with self.assertRaises(ValueError):
            load_final_test(self.data_dir, draft)
        self.assertFalse((self.data_dir / USAGE_LOG).exists())

    def test_selection_must_match_the_data(self):
        self.freeze()
        other = make_dataset(self.root / "outro", small_config(seed=99))
        with self.assertRaises(ValueError):
            load_final_test(other, self.root / "selecao-congelada.json")
        self.assertFalse((other / USAGE_LOG).exists())

    def test_use_is_logged_and_unique(self):
        self.freeze()
        selection = self.root / "selecao-congelada.json"
        test, entry = load_final_test(self.data_dir, selection)
        self.assertEqual(test["teste_final_u"].shape, (4, 400))
        lines = (self.data_dir / USAGE_LOG).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        logged = json.loads(lines[0])
        self.assertEqual(logged["selecao_sha256"], entry["selecao_sha256"])
        self.assertEqual(logged["chamador"], "test_protocolo_v2.py:test_use_is_logged_and_unique")
        self.assertEqual(logged["teste_final_sha256"], read_json(self.data_dir / MANIFEST_FILE)
                         ["arquivos_sha256"][TEST_FILE])
        with self.assertRaises(PermissionError):
            load_final_test(self.data_dir, selection)
        load_final_test(self.data_dir, selection, justification="falha técnica antes das métricas")
        lines = (self.data_dir / USAGE_LOG).read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(l)["uso"] for l in lines], [1, 2])
        self.assertEqual(json.loads(lines[1])["justificativa"], "falha técnica antes das métricas")

    def test_changed_code_or_artifact_blocks_the_test(self):
        code = self.root / "codigo"
        code.mkdir()
        (code / "modelo.py").write_text("x = 1\n", encoding="utf-8")
        self.freeze(code_files=["modelo.py"], code_root=code)
        (code / "modelo.py").write_text("x = 2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_final_test(self.data_dir, self.root / "selecao-congelada.json", code_root=code)
        (code / "modelo.py").write_text("x = 1\n", encoding="utf-8")
        (self.root / "modelo-arx.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_final_test(self.data_dir, self.root / "selecao-congelada.json", code_root=code)
        self.assertFalse((self.data_dir / USAGE_LOG).exists())

    def test_frozen_selection_is_not_overwritten(self):
        self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()


class FinalEvaluationChecks(unittest.TestCase):
    """avaliar_teste_final.py de ponta a ponta, com persistência e ARX."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = small_config(noise=0.01)
        self.data_dir = make_dataset(self.root, self.config)
        development, _ = load_development(self.data_dir)
        self.arx = fit_arx(development["treino_u"], development["treino_y"], 4, 4)
        write_json(self.root / "modelo-arx.json", self.arx.to_dict())
        self.selection = self.root / "selecao-congelada.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_evaluates_once_with_intervals(self):
        freeze_selection(self.selection, scenario="teste", data_dir=self.data_dir, families=reference_families(),
                         best_traditional="arx", comparisons=[("arx", "persistencia")])
        out = avaliar_teste_final.run(self.selection, self.root / "avaliacao")
        for name in ("metricas-teste-final.csv", "metricas-teste-final-sem-ruido.csv", "intervalos-bootstrap.json",
                     "avaliacao-teste-final.json", "execucao.json", "resumo.md", "uso-teste-final.json"):
            self.assertTrue((out / name).exists(), name)
        lines = (self.data_dir / USAGE_LOG).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["chamador"], "avaliar_teste_final.py:run")

        with np.load(self.data_dir / TEST_FILE) as f:  # conferência independente, só neste teste
            u, y = f["teste_final_u"], f["teste_final_y"]
        settings = evaluation_settings(self.config)
        expected = {"arx": evaluate(self.arx, u, y, settings), "persistencia": evaluate(Persistence(), u, y, settings)}
        stored = read_json(out / "avaliacao-teste-final.json")
        for family in ("arx", "persistencia"):
            self.assertEqual(stored[family][0]["rows"], json.loads(json.dumps(expected[family]["rows"])))

        intervals = read_json(out / "intervalos-bootstrap.json")["intervals"]
        main = intervals["medida/trajectory/50"]
        for family in main["families"].values():
            self.assertLessEqual(family["interval"][0], family["point"])
            self.assertLessEqual(family["point"], family["interval"][1])
        self.assertEqual(main["families"]["arx"]["point"], expected["arx"]["rows"][-1]["rmse_trajectory_m"])
        self.assertTrue(main["comparisons"][0]["pre_registrada"])
        self.assertEqual(main["comparisons"][0]["verdict"], "a_menor")
        self.assertIn("sem_ruido/trajectory/50", intervals)

        with self.assertRaises(PermissionError):
            avaliar_teste_final.run(self.selection, self.root / "avaliacao-2")

    def test_adapter_failure_does_not_consume_the_test(self):
        @register_adapter("falha_teste")
        def failing(entry, member, context):
            raise RuntimeError("adaptador quebrado")
        try:
            families = reference_families()
            families["quebrado"] = {**copy.deepcopy(families["persistencia"]), "adaptador": "falha_teste"}
            freeze_selection(self.selection, scenario="teste", data_dir=self.data_dir, families=families,
                             best_traditional="arx")
            with self.assertRaises(RuntimeError):
                avaliar_teste_final.run(self.selection, self.root / "avaliacao")
            self.assertFalse((self.data_dir / USAGE_LOG).exists())
            self.assertFalse((self.root / "avaliacao").exists())
        finally:
            ADAPTERS.pop("falha_teste")


if __name__ == "__main__":
    unittest.main()
