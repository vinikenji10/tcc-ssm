"""Identificação por entrada/saída: planta de Duffing (mola cúbica), persistência, ARX e NARX polinomial.

Mesma convenção, divisão e avaliação do E001: y[t] é medido antes de aplicar u[t]; u[t] gera y[t+1].
Muda a planta (rigidez cúbica) e a excitação (amplitude aleatória por patamar, APRBS), necessária para
excitar a não linearidade em vários níveis. Estados físicos ficam fora do ajuste e da avaliação.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import scipy

from experimento_linear import (Persistence, evaluate, evaluation_origins, normalization_from_training, select_arx,
                                summarize_rollouts, write_csv, write_json)


def validate_config(config):
    p, d, e = (config[key] for key in ("plant", "data", "evaluation"))
    if min(p["mass"], p["stiffness"], p["sample_time"]) <= 0 or min(p["damping"], p["cubic_stiffness"]) < 0:
        raise ValueError("Parâmetros físicos inválidos.")
    if p["rk4_substeps"] < 1:
        raise ValueError("Use pelo menos um subpasso de integração.")
    if min(d[k] for k in ("train_episodes", "validation_episodes", "test_episodes")) < 1:
        raise ValueError("Cada conjunto precisa de pelo menos uma trajetória.")
    if d["hold_min"] < 1 or d["hold_max"] < d["hold_min"]:
        raise ValueError("Intervalos de retenção inválidos.")
    orders = config["arx"]["output_orders"] + config["arx"]["input_orders"] + config["narx"]["orders"]
    if min(orders) < 1 or min(config["narx"]["degrees"]) < 1:
        raise ValueError("Ordens e graus devem ser positivos.")
    if e["warmup"] < max(orders):
        raise ValueError("Aquecimento insuficiente para as ordens propostas.")
    if d["steps"] < e["warmup"] + max(e["horizons"]):
        raise ValueError("Trajetórias curtas demais para a avaliação.")


def duffing_step(x, u, plant):
    """Um período de amostragem com u constante (ZOH), RK4 com subpassos. x: (..., 2) = [posição, velocidade]."""
    m, c, k, k3 = (plant[key] for key in ("mass", "damping", "stiffness", "cubic_stiffness"))
    h = plant["sample_time"] / plant["rk4_substeps"]

    def f(state):
        p, v = state[..., 0], state[..., 1]
        return np.stack([v, (u - c * v - k * p - k3 * p ** 3) / m], axis=-1)

    for _ in range(plant["rk4_substeps"]):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3_ = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3_)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3_ + k4)
    return x


def generate_dataset(config):
    validate_config(config)
    d = config["data"]
    split_sizes = [d[key] for key in ("train_episodes", "validation_episodes", "test_episodes")]
    split_names = ("treino", "validacao", "teste")
    children = np.random.SeedSequence(config["seed"]).spawn(sum(split_sizes))
    signals, physical_states, manifest = {}, {}, []
    cursor = 0
    for split, count in zip(split_names, split_sizes):
        u = np.empty((count, d["steps"]), dtype=np.float64)
        x = np.empty((count, d["steps"] + 1, 2), dtype=np.float64)
        noise = np.empty((count, d["steps"] + 1), dtype=np.float64)
        ids = []
        for episode in range(count):
            child = children[cursor]
            rng = np.random.default_rng(child)
            episode_id = f"{split}-{episode:03d}"
            ids.append(episode_id)
            x[episode, 0] = [rng.uniform(-d["initial_position_limit"], d["initial_position_limit"]),
                             rng.uniform(-d["initial_velocity_limit"], d["initial_velocity_limit"])]
            t = 0
            while t < d["steps"]:
                hold = int(rng.integers(d["hold_min"], d["hold_max"] + 1))
                u[episode, t:min(t + hold, d["steps"])] = rng.uniform(-d["input_amplitude"], d["input_amplitude"])
                t += hold
            noise[episode] = rng.normal(0, d["measurement_noise_std"], d["steps"] + 1)
            manifest.append({"id": episode_id, "split": split, "spawn_key": list(child.spawn_key),
                             "seed_words": child.generate_state(4).tolist()})
            cursor += 1
        for t in range(d["steps"]):  # todas as trajetórias do conjunto em paralelo
            x[:, t + 1] = duffing_step(x[:, t], u[:, t], config["plant"])
        signals[f"{split}_u"] = u
        signals[f"{split}_y"] = x[:, :, 0] + noise
        signals[f"{split}_ids"] = np.array(ids)
        physical_states[f"{split}_x"] = x
    return signals, physical_states, manifest


def clean_metrics(result, y_clean, settings):
    """Mesmas previsões, comparadas com a posição sem ruído. Só para relatar; nada é escolhido com isso."""
    hmax = max(settings["horizons"])
    origins = evaluation_origins(y_clean.shape[1] - 1, settings)
    targets = np.stack([y_clean[e, t + 1:t + hmax + 1] for e in range(len(y_clean)) for t in origins])
    first = result["rows"][0] if result["rows"] else {"model": None, "parameters": None}
    return summarize_rollouts(first["model"], first["parameters"], result["prediction"], targets, settings, len(origins))


def monomials(n_variables, degree):
    """Índices de todos os monômios de grau até `degree`; o índice n_variables vale 1 (termo constante)."""
    return np.array(list(combinations_with_replacement(range(n_variables + 1), degree)))


class PolynomialNARX:
    """y[t+1] = soma de monômios de y[t..t-n+1] e u[t..t-n+1] até o grau escolhido, incluindo constante."""

    def __init__(self, order, degree, coefficients, diagnostics=None):
        self.order, self.degree = order, degree
        self.terms = monomials(2 * order, degree)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.diagnostics = diagnostics or {}
        self.name = f"NARX({order},{order}) grau {degree}"
        self.parameters = len(self.coefficients)

    def predict_step(self, u_history, y_history):
        z = np.r_[y_history[-self.order:][::-1], u_history[-self.order:][::-1], 1.0]
        return float(np.prod(z[self.terms], axis=1) @ self.coefficients)

    def to_dict(self):
        return {"name": self.name, "order": self.order, "degree": self.degree,
                "coefficients": self.coefficients.tolist(), "parameters": self.parameters,
                "terms": self.terms.tolist(),
                "variable_order": "y[t], y[t-1], ...; u[t], u[t-1], ...; constante (último índice)",
                "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, spec):
        return cls(spec["order"], spec["degree"], spec["coefficients"], spec.get("diagnostics"))


def fit_narx(u, y, order, degree):
    """Mínimos quadrados do erro de um passo, com colunas escaladas só para o condicionamento."""
    t = np.arange(order - 1, u.shape[1])
    z = np.stack([y[:, t - lag] for lag in range(order)] + [u[:, t - lag] for lag in range(order)]
                 + [np.ones((len(u), len(t)))], axis=-1).reshape(-1, 2 * order + 1)
    terms = monomials(2 * order, degree)
    design = np.prod(z[:, terms], axis=2)
    scale = design.std(axis=0)
    scale[scale == 0] = 1.0
    target = y[:, t + 1].reshape(-1)
    scaled, _, rank, singular_values = np.linalg.lstsq(design / scale, target, rcond=None)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None
    return PolynomialNARX(order, degree, scaled / scale,
                          {"rank": int(rank), "columns": design.shape[1], "condition_number_scaled": condition,
                           "fitting_samples": len(target)})


def select_narx(signals, config):
    candidates, models = [], []
    for order in config["narx"]["orders"]:
        for degree in config["narx"]["degrees"]:
            start = perf_counter()
            model = fit_narx(signals["treino_u"], signals["treino_y"], order, degree)
            fitting_seconds = perf_counter() - start
            result = evaluate(model, signals["validacao_u"], signals["validacao_y"], config["evaluation"])
            score = result["rows"][-1]["rmse_trajectory_m"] if result["rows"] else None
            candidates.append({"model": model.name, "order": order, "degree": degree,
                               "validation_rmse_trajectory_m": score,
                               "nonfinite_values": result["nonfinite_values"],
                               "fitting_seconds": fitting_seconds, "parameters": model.parameters,
                               **model.diagnostics})
            models.append(model)
    valid = [i for i, c in enumerate(candidates) if c["validation_rmse_trajectory_m"] is not None]
    if not valid:
        raise RuntimeError("Nenhum candidato NARX produziu previsões finitas na validação.")
    best = min(candidates[i]["validation_rmse_trajectory_m"] for i in valid)
    tied = [i for i in valid if candidates[i]["validation_rmse_trajectory_m"] <= best + config["narx"]["numerical_tie_rmse"]]
    selected = min(tied, key=lambda i: (models[i].parameters, models[i].order, models[i].degree))
    return models[selected], candidates, candidates[selected]["fitting_seconds"]


def nonlinearity_diagnostics(states, plant):
    """Quanto da força restauradora vem do termo cúbico, nas posições do treino."""
    p = states["treino_x"][:, :, 0]
    linear, cubic = plant["stiffness"] * p, plant["cubic_stiffness"] * p ** 3
    return {"position_min_m": float(p.min()), "position_max_m": float(p.max()),
            "position_std_m": float(p.std()),
            "rms_linear_force_N": float(np.sqrt(np.mean(linear ** 2))),
            "rms_cubic_force_N": float(np.sqrt(np.mean(cubic ** 2))),
            "cubic_to_linear_rms_ratio": float(np.sqrt(np.mean(cubic ** 2)) / np.sqrt(np.mean(linear ** 2))),
            "max_cubic_to_linear_ratio": float(plant["cubic_stiffness"] * p.max() ** 2 / plant["stiffness"])}


def make_plots(out, signals, states, references, results, config):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "figure.dpi": 120,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    colors = {"real": "#112D4E", "arx": "#3F72AF", "narx": "#B91C1C", "persistence": "#D97706"}
    figures = out / "figuras"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    def color(name):
        return colors["persistence"] if name == "Persistência" else colors["narx"] if name.startswith("NARX") else colors["arx"]

    plant, dt = config["plant"], config["plant"]["sample_time"]
    u, y = signals["teste_u"][0], signals["teste_y"][0]
    n = min(500, len(u))
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    ax[0].step(np.arange(n) * dt, u[:n], where="post", color=colors["real"])
    ax[0].set(ylabel="Força u [N]", title="Trajetória de teste: oscilador de Duffing (mola cúbica)")
    ax[1].plot(np.arange(n + 1) * dt, y[:n + 1], color=colors["real"])
    ax[1].set(xlabel="Tempo [s]", ylabel="Posição y [m]")
    save(fig, "01-entrada-saida")

    p_train = states["treino_x"][:, :, 0]
    p = np.linspace(p_train.min(), p_train.max(), 200)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax[0].plot(p, plant["stiffness"] * p, "--", color=colors["arx"], label="Mola linear, k·p")
    ax[0].plot(p, plant["stiffness"] * p + plant["cubic_stiffness"] * p ** 3, color=colors["real"],
               label="Duffing, k·p + k₃·p³")
    ax[0].set(xlabel="Posição p [m]", ylabel="Força restauradora [N]", title="Rigidez na faixa de posições do treino")
    ax[0].legend()
    ax[1].hist(p_train.ravel(), bins=60, color=colors["real"], alpha=0.8)
    ax[1].set(xlabel="Posição p [m]", ylabel="Amostras", title="Posições visitadas no treino")
    save(fig, "02-nao-linearidade")

    horizon = max(config["evaluation"]["horizons"])
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for name, result in results.items():
        if result["rmse_curve"] is not None:
            ax.plot(np.arange(1, horizon + 1), result["rmse_curve"], color=color(name), label=name)
    ax.set(yscale="log", xlabel="Passo futuro", ylabel="RMSE no ponto final [m]", title="Erro por horizonte no teste")
    ax.legend()
    save(fig, "03-erro-por-horizonte")


def run(config_path, output_path):
    config_path, out = Path(config_path), Path(output_path)
    if out.exists():
        raise FileExistsError(f"A pasta {out} já existe. Escolha outra saída para preservar a execução.")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    config["evaluation"]["horizons"] = sorted(set(config["evaluation"]["horizons"]))
    out.mkdir(parents=True)
    try:
        start = perf_counter()
        signals, states, manifest = generate_dataset(config)
        generation_seconds = perf_counter() - start
        np.savez_compressed(out / "dados.npz", **signals)
        diagnostics = out / "diagnostico"
        diagnostics.mkdir()
        np.savez_compressed(diagnostics / "estados-fisicos.npz", **states)
        nonlinearity = nonlinearity_diagnostics(states, config["plant"])
        write_json(diagnostics / "nao-linearidade.json", nonlinearity)
        write_json(out / "config.json", config)
        write_json(out / "manifesto-trajetorias.json", manifest)
        write_json(out / "normalizacao-treino.json", normalization_from_training(signals))

        arx, arx_selection, arx_seconds = select_arx(signals, config)
        narx, narx_selection, narx_seconds = select_narx(signals, config)
        write_json(out / "selecao-validacao.json", {"criterion": "RMSE da trajetória no maior horizonte",
                   "tie_rule": "menor número de parâmetros dentro da tolerância numérica",
                   "selected_arx": arx.name, "selected_narx": narx.name,
                   "arx_candidates": arx_selection, "narx_candidates": narx_selection})
        write_json(out / "modelo-arx.json", arx.to_dict())
        write_json(out / "modelo-narx.json", narx.to_dict())

        # Teste: lido só agora, depois das escolhas.
        results = {}
        for model in (Persistence(), arx, narx):
            begin = perf_counter()
            results[model.name] = evaluate(model, signals["teste_u"], signals["teste_y"], config["evaluation"])
            results[model.name]["evaluation_seconds"] = perf_counter() - begin
        rows = [row for result in results.values() for row in result["rows"]]
        write_csv(out / "metricas-teste.csv", rows)
        noisy = config["data"]["measurement_noise_std"] > 0
        if noisy:  # saída sem ruído só para relatar o erro em relação à planta; fica fora dos dados dos modelos
            np.savez_compressed(diagnostics / "saida-sem-ruido.npz",
                                **{f"{split}_y": states[f"{split}_x"][:, :, 0] for split in ("treino", "validacao", "teste")})
            clean_rows = [row for result in results.values()
                          for row in clean_metrics(result, states["teste_x"][:, :, 0], config["evaluation"])["rows"]]
            write_csv(out / "metricas-teste-sem-ruido.csv", clean_rows)
        write_json(out / "avaliacao-teste.json", {name: {k: v for k, v in result.items() if k not in ("prediction", "target")}
                                                for name, result in results.items()})
        make_plots(out, signals, states, (arx, narx), results, config)
        import matplotlib
        write_json(out / "execucao.json", {
            "status": "concluido", "created_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            "scope": "planta de Duffing simulada, persistência, ARX e NARX polinomial; Mamba-3 em execução separada",
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data_sha256": hashlib.sha256((out / "dados.npz").read_bytes()).hexdigest(),
            "generation_seconds": generation_seconds,
            "selected_arx_fitting_seconds": arx_seconds, "selected_narx_fitting_seconds": narx_seconds,
            "test_evaluation_seconds": {name: r["evaluation_seconds"] for name, r in results.items()},
            "hardware_regime": "CPU; tempos de ajuste e de avaliação em Python, sem benchmark"})

        text = ["# Referências clássicas na planta de Duffing", "",
                "Execução local concluída. Oscilador de Duffing simulado (mola cúbica), "
                + ("sem ruído." if config["data"]["measurement_noise_std"] == 0 else "com ruído de medição conforme configuração."), "",
                f"Não linearidade nas posições do treino: força cúbica / força linear = {nonlinearity['cubic_to_linear_rms_ratio']:.2f} (RMS), "
                f"até {nonlinearity['max_cubic_to_linear_ratio']:.1f} na maior posição ({nonlinearity['position_max_m']:.3f} m).", "",
                f"Escolhidos pela validação: **{arx.name}** ({arx.parameters} parâmetros) e **{narx.name}** ({narx.parameters} parâmetros).", "",
                "| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |",
                "|---|---:|---:|---:|---:|"]
        for row in rows:
            text.append(f"| {row['model']} | {row['horizon_steps']} | {row['rmse_endpoint_m']:.3e} | "
                        f"{row['rmse_trajectory_m']:.3e} | {row['nrmse_trajectory']:.3e} |")
        if noisy:
            text += ["", "Tabela acima: erro em relação à posição medida, com ruído. Abaixo, as mesmas previsões "
                     "comparadas com a posição sem ruído (`metricas-teste-sem-ruido.csv`):", "",
                     "| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] |", "|---|---:|---:|---:|"]
            for row in clean_rows:
                text.append(f"| {row['model']} | {row['horizon_steps']} | {row['rmse_endpoint_m']:.3e} | {row['rmse_trajectory_m']:.3e} |")
        text += ["", "## Artefatos", "", "- `dados.npz`, `normalizacao-treino.json`: sinais e estatísticas do treino.",
                 "- `selecao-validacao.json`, `modelo-arx.json`, `modelo-narx.json`: candidatos, escolha e coeficientes.",
                 "- `metricas-teste.csv`, `avaliacao-teste.json`: resultados no teste.",
                 "- `figuras/`: entrada e saída, não linearidade e erro por horizonte.",
                 "- `diagnostico/`: estados físicos e medida da não linearidade; fora dos modelos.", ""]
        (out / "resumo.md").write_text("\n".join(text))
        print("\n".join(text))
        return out
    except Exception as error:
        write_json(out / "execucao.json", {"status": "falhou", "error": str(error)})
        raise


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "experimentos/config-duffing.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
