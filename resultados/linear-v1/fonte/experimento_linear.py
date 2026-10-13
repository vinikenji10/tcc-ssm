"""Identificação por entrada/saída: planta linear, persistência e ARX.

Convenção: y[t] é medido antes de aplicar u[t]; u[t] gera y[t+1].
Estados físicos ficam em um arquivo de diagnóstico, fora do ajuste e da avaliação.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from scipy.linalg import expm


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_config(config):
    p, d, e, a = (config[key] for key in ("plant", "data", "evaluation", "arx"))
    if min(p["mass"], p["stiffness"], p["sample_time"]) <= 0 or p["damping"] < 0:
        raise ValueError("Parâmetros físicos inválidos.")
    if min(d[k] for k in ("train_episodes", "validation_episodes", "test_episodes")) < 1:
        raise ValueError("Cada conjunto precisa de pelo menos uma trajetória.")
    if d["hold_min"] < 1 or d["hold_max"] < d["hold_min"]:
        raise ValueError("Intervalos de retenção inválidos.")
    if min(d["input_amplitude"], d["initial_position_limit"], d["initial_velocity_limit"], d["measurement_noise_std"]) < 0:
        raise ValueError("Amplitudes e desvios devem ser não negativos.")
    if not e["horizons"] or min(e["horizons"]) < 1 or e["origin_stride"] < 1:
        raise ValueError("Horizontes e passo de avaliação devem ser positivos.")
    if min(a["output_orders"] + a["input_orders"]) < 1:
        raise ValueError("As ordens ARX devem ser positivas.")
    if e["warmup"] < max(a["output_orders"] + a["input_orders"]):
        raise ValueError("Aquecimento insuficiente para as ordens propostas.")
    if d["steps"] < e["warmup"] + max(e["horizons"]):
        raise ValueError("Trajetórias curtas demais para a avaliação.")


def discretize(plant):
    """ZOH exata via exponencial da matriz aumentada, sem inverter A."""
    m, c, k, dt = (plant[key] for key in ("mass", "damping", "stiffness", "sample_time"))
    augmented = np.array([[0.0, 1.0, 0.0], [-k / m, -c / m, 1.0 / m], [0.0, 0.0, 0.0]])
    transition = expm(augmented * dt)
    return transition[:2, :2], transition[:2, 2]


def generate_dataset(config):
    validate_config(config)
    d = config["data"]
    ad, bd = discretize(config["plant"])
    split_sizes = [d[key] for key in ("train_episodes", "validation_episodes", "test_episodes")]
    split_names = ("treino", "validacao", "teste")
    children = np.random.SeedSequence(config["seed"]).spawn(sum(split_sizes))
    signals, physical_states, manifest = {}, {}, []
    cursor = 0
    for split, count in zip(split_names, split_sizes):
        u = np.empty((count, d["steps"]), dtype=np.float64)
        x = np.empty((count, d["steps"] + 1, 2), dtype=np.float64)
        y = np.empty((count, d["steps"] + 1), dtype=np.float64)
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
                value = d["input_amplitude"] * rng.choice([-1, 1])
                u[episode, t:min(t + hold, d["steps"])] = value
                t += hold
            for t in range(d["steps"]):
                x[episode, t + 1] = ad @ x[episode, t] + bd * u[episode, t]
            y[episode] = x[episode, :, 0] + rng.normal(0, d["measurement_noise_std"], d["steps"] + 1)
            manifest.append({"id": episode_id, "split": split, "spawn_key": list(child.spawn_key),
                             "seed_words": child.generate_state(4).tolist()})
            cursor += 1
        signals[f"{split}_u"] = u
        signals[f"{split}_y"] = y
        signals[f"{split}_ids"] = np.array(ids)
        physical_states[f"{split}_x"] = x
    return signals, physical_states, manifest, ad, bd


def normalization_from_training(signals):
    # Mesma estatística para entrada y e alvo; calculada somente em medições de treino.
    return {name: {"mean": float(signals[f"treino_{name}"].mean()),
                   "std": float(signals[f"treino_{name}"].std())}
            for name in ("u", "y")}


class Persistence:
    name = "Persistência"
    parameters = 0

    def predict_step(self, u_history, y_history):
        return float(y_history[-1])


class ARX:
    def __init__(self, output_order, input_order, coefficients, diagnostics=None):
        self.output_order = output_order
        self.input_order = input_order
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.diagnostics = diagnostics or {}
        self.name = f"ARX({output_order},{input_order})"
        self.parameters = len(self.coefficients)

    def predict_step(self, u_history, y_history):
        features = np.r_[y_history[-self.output_order:][::-1],
                         u_history[-self.input_order:][::-1], 1.0]
        return float(features @ self.coefficients)

    def to_dict(self):
        return {"name": self.name, "output_order": self.output_order, "input_order": self.input_order,
                "coefficients": self.coefficients.tolist(), "parameters": self.parameters,
                "feature_order": "y[t], y[t-1], ...; u[t], u[t-1], ...; intercepto",
                "diagnostics": self.diagnostics}


def fit_arx(u, y, output_order, input_order):
    """Ajuste usa somente os sinais fornecidos; nenhum parâmetro da planta."""
    start = max(output_order, input_order) - 1
    t = np.arange(start, u.shape[1])
    blocks = [y[:, t - lag] for lag in range(output_order)]
    blocks += [u[:, t - lag] for lag in range(input_order)]
    blocks += [np.ones((len(u), len(t)))]
    design = np.stack(blocks, axis=-1).reshape(-1, len(blocks))
    target = y[:, t + 1].reshape(-1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None
    return ARX(output_order, input_order, coefficients,
               {"rank": int(rank), "columns": design.shape[1], "condition_number": condition,
                "fitting_samples": len(target)})


def rollout(model, u, observed_prefix, origin, horizon):
    """Somente o prefixo medido até a origem entra no preditor.

    A sequência futura de u é conhecida; futuras medições de y não são aceitas.
    Cada origem reinicia seu histórico local e realimenta as próprias previsões.
    """
    if len(observed_prefix) != origin + 1:
        raise ValueError("Forneça exclusivamente y[:origin+1], sem medições futuras.")
    if origin < 0 or horizon < 1 or origin + horizon > len(u):
        raise ValueError("Origem/horizonte fora do intervalo de dados.")
    y_history = np.empty(origin + horizon + 1, dtype=np.float64)
    y_history[:origin + 1] = observed_prefix
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(horizon):
            t = origin + j
            y_history[t + 1] = model.predict_step(u[:t + 1], y_history[:t + 1])
    return y_history[origin + 1:]


def evaluate(model, u, y, settings):
    hmax = max(settings["horizons"])
    origins = np.arange(settings["warmup"], u.shape[1] - hmax + 1, settings["origin_stride"])
    predictions, targets = [], []
    for episode in range(len(u)):
        for t in origins:
            predictions.append(rollout(model, u[episode], y[episode, :t + 1], int(t), hmax))
            targets.append(y[episode, t + 1:t + hmax + 1])
    predictions, targets = np.array(predictions), np.array(targets)
    errors = predictions - targets
    finite = np.isfinite(errors)
    if not finite.all():
        return {"origins_per_episode": len(origins), "rollouts": len(predictions),
                "nonfinite_values": int((~finite).sum()), "rows": [], "rmse_curve": None,
                "prediction": predictions, "target": targets}
    # Um denominador comum em todos os horizontes, calculado só para expressar a métrica.
    target_std = float(targets.std())
    curve = np.sqrt(np.mean(errors ** 2, axis=0))
    rows = []
    for horizon in settings["horizons"]:
        end = float(curve[horizon - 1])
        trajectory = float(np.sqrt(np.mean(errors[:, :horizon] ** 2)))
        rows.append({"model": model.name, "horizon_steps": horizon,
                     "rmse_endpoint_m": end, "rmse_trajectory_m": trajectory,
                     "nrmse_endpoint": end / target_std if target_std else None,
                     "nrmse_trajectory": trajectory / target_std if target_std else None,
                     "normalization_std_m": target_std, "rollouts": len(predictions),
                     "nonfinite_values": 0, "parameters": model.parameters})
    return {"origins_per_episode": len(origins), "rollouts": len(predictions), "nonfinite_values": 0,
            "rows": rows, "rmse_curve": curve.tolist(), "prediction": predictions, "target": targets}


def select_arx(signals, config):
    candidates, models = [], []
    for ny in config["arx"]["output_orders"]:
        for nu in config["arx"]["input_orders"]:
            start = perf_counter()
            model = fit_arx(signals["treino_u"], signals["treino_y"], ny, nu)
            fitting_seconds = perf_counter() - start
            result = evaluate(model, signals["validacao_u"], signals["validacao_y"], config["evaluation"])
            score = result["rows"][-1]["rmse_trajectory_m"] if result["rows"] else None
            candidates.append({"model": model.name, "output_order": ny, "input_order": nu,
                               "validation_rmse_trajectory_m": score,
                               "nonfinite_values": result["nonfinite_values"],
                               "fitting_seconds": fitting_seconds, "parameters": model.parameters,
                               **model.diagnostics})
            models.append(model)
    valid = [i for i, c in enumerate(candidates) if c["validation_rmse_trajectory_m"] is not None]
    if not valid:
        raise RuntimeError("Nenhum candidato ARX produziu previsões finitas na validação.")
    best = min(candidates[i]["validation_rmse_trajectory_m"] for i in valid)
    tied = [i for i in valid if candidates[i]["validation_rmse_trajectory_m"] <= best + config["arx"]["numerical_tie_rmse"]]
    selected = min(tied, key=lambda i: (models[i].parameters, models[i].output_order, models[i].input_order))
    return models[selected], candidates, candidates[selected]["fitting_seconds"]


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out, signals, model, results, config):
    # Cache local, sem modificar configurações do usuário.
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "figure.dpi": 120,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    colors = {"real": "#112D4E", "arx": "#3F72AF", "persistence": "#D97706"}
    figures = out / "figuras"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    u, y = signals["teste_u"][0], signals["teste_y"][0]
    dt = config["plant"]["sample_time"]
    n = min(500, len(u))
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    ax[0].step(np.arange(n) * dt, u[:n], where="post", color=colors["real"])
    ax[0].set(ylabel="Força u [N]", title="Trajetória de teste: massa–mola–amortecedor, sem ruído" if config["data"]["measurement_noise_std"] == 0 else "Trajetória de teste: massa–mola–amortecedor")
    ax[1].plot(np.arange(n + 1) * dt, y[:n + 1], color=colors["real"])
    ax[1].set(xlabel="Tempo [s]", ylabel="Posição y [m]")
    save(fig, "01-entrada-saida")

    origin = config["evaluation"]["warmup"]
    n = min(250, len(u) - origin)
    actual = y[origin + 1:origin + n + 1]
    predicted = np.array([model.predict_step(u[:t + 1], y[:t + 1]) for t in range(origin, origin + n)])
    persist = y[origin:origin + n]
    time = (np.arange(n) + origin + 1) * dt
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    ax[0].plot(time, actual, color=colors["real"], label="Medida")
    ax[0].plot(time, predicted, "--", color=colors["arx"], label=model.name)
    ax[0].plot(time, persist, color=colors["persistence"], alpha=0.65, label="Persistência")
    ax[0].set(ylabel="Posição [m]", title="Previsão de um passo com histórico medido real")
    ax[0].legend(ncols=3)
    ax[1].plot(time, predicted - actual, color=colors["arx"], label="Erro ARX")
    ax[1].plot(time, persist - actual, color=colors["persistence"], label="Erro persistência")
    ax[1].set(xlabel="Tempo [s]", ylabel="Previsto − medido [m]")
    save(fig, "02-previsao-um-passo")

    horizon = max(config["evaluation"]["horizons"])
    steps = np.arange(horizon + 1)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(steps * dt, y[origin:origin + horizon + 1], color=colors["real"], lw=2, label="Medida")
    for baseline, color in [(model, colors["arx"]), (Persistence(), colors["persistence"])]:
        values = np.r_[y[origin], rollout(baseline, u, y[:origin + 1], origin, horizon)]
        ax.plot(steps * dt, values, "--", color=color, label=baseline.name)
    ax.axvline(0, color="#6B7280", lw=1)
    ax.set(xlabel="Tempo após a origem [s]", ylabel="Posição [m]",
           title=f"Simulação de {horizon} passos: saídas futuras reais não entram no modelo")
    ax.legend()
    save(fig, "03-simulacao-multiplos-passos")

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for name, result in results.items():
        if result["rmse_curve"] is not None:
            color = colors["persistence"] if name == "Persistência" else colors["arx"]
            ax.plot(np.arange(1, horizon + 1), result["rmse_curve"], color=color, label=name)
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set(xlabel="Passo futuro", ylabel="RMSE no ponto final [m]",
           title="Erro por horizonte no teste | escala linear abaixo de 10⁻¹² m")
    ax.legend()
    save(fig, "04-erro-por-horizonte")


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
        signals, states, manifest, ad, bd = generate_dataset(config)
        np.savez_compressed(out / "dados.npz", **signals)
        diagnostics = out / "diagnostico"
        diagnostics.mkdir()
        np.savez_compressed(diagnostics / "estados-fisicos.npz", **states)
        write_json(diagnostics / "planta-discreta.json", {"A_d": ad.tolist(), "B_d": bd.tolist()})
        write_json(out / "config.json", config)
        write_json(out / "manifesto-trajetorias.json", manifest)
        write_json(out / "normalizacao-treino.json", normalization_from_training(signals))
        model, selection, fitting_seconds = select_arx(signals, config)
        write_json(out / "selecao-validacao.json", {"criterion": "RMSE da trajetória no maior horizonte",
                   "tie_rule": "menor número de parâmetros dentro da tolerância numérica",
                   "selected": model.name, "candidates": selection})
        write_json(out / "modelo-arx.json", model.to_dict())
        results = {baseline.name: evaluate(baseline, signals["teste_u"], signals["teste_y"], config["evaluation"])
                   for baseline in (Persistence(), model)}
        rows = [row for result in results.values() for row in result["rows"]]
        if rows:
            write_csv(out / "metricas-teste.csv", rows)
        write_json(out / "avaliacao-teste.json", {name: {k: v for k, v in result.items() if k not in ("prediction", "target")}
                                                for name, result in results.items()})
        make_plots(out, signals, model, results, config)
        import matplotlib
        metadata = {"status": "concluido", "created_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
                    "scope": "planta linear simulada, persistência e ARX; Mamba-3 e MPC não executados",
                    "python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                    "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "data_sha256": hashlib.sha256((out / "dados.npz").read_bytes()).hexdigest(),
                    "selected_arx_fitting_seconds": fitting_seconds,
                    "hardware_regime": "CPU; tempo do ajuste, sem benchmark de inferência"}
        write_json(out / "execucao.json", metadata)
        text = ["# Primeiro experimento de identificação", "", "Execução local concluída. Planta linear simulada; sinais sem ruído." if config["data"]["measurement_noise_std"] == 0 else "Execução local concluída. Planta linear simulada; ruído de medição conforme configuração.",
                "", "Mamba-3 e MPC ainda não foram executados.", "",
                f"ARX selecionado somente pela validação: **{model.name}**, com {model.parameters} parâmetros.",
                "", "| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |",
                "|---|---:|---:|---:|---:|"]
        for row in rows:
            nrmse = f"{row['nrmse_trajectory']:.3e}" if row["nrmse_trajectory"] is not None else "indefinido"
            text.append(f"| {row['model']} | {row['horizon_steps']} | {row['rmse_endpoint_m']:.3e} | {row['rmse_trajectory_m']:.3e} | {nrmse} |")
        text += ["", "## O que esse resultado estabelece", "",
                 "O ARX foi ajustado apenas aos sinais de treino e avaliado em trajetórias independentes. Na simulação de vários passos, recebe o histórico observado até a origem e, depois dela, usa exclusivamente suas próprias saídas previstas e as entradas conhecidas.", "",
                 "No caso linear ideal e sem ruído, uma referência ARX de ordem adequada pode reproduzir a dinâmica até a precisão numérica. Erros nessa faixa devem ser descritos como próximos da precisão numérica, sem interpretar diferenças ínfimas entre ordens como ganhos práticos.", "",
                 "O experimento verifica a base de simulação e avaliação. Ele não demonstra desempenho de Mamba-3, superioridade de SSMs neurais, robustez a ruído, identificação não linear ou controle em malha fechada.", "",
                 "## Artefatos", "", "- `dados.npz`: sinais e identificadores separados por trajetória.",
                 "- `normalizacao-treino.json`: estatísticas calculadas somente no treino.",
                 "- `modelo-arx.json` e `selecao-validacao.json`: ajuste e escolha sem usar o teste.",
                 "- `metricas-teste.csv` e `avaliacao-teste.json`: resultados e curva por horizonte.",
                 "- `figuras/`: quatro figuras, em PNG e SVG.",
                 "- `diagnostico/`: estados físicos e discretização; não utilizados como atributos dos modelos.",
                 "- `execucao.json`: versões, hashes e escopo verificado.", "",
                 "## Próximo passo", "", "Adicionar o preditor Mamba-3 aos mesmos sinais e origens de avaliação, após validar seu ambiente. Antes de usar redes com memória, implementar e conferir o aquecimento e a regra de reinício de estado. Não comparar uma rede avaliada com medições futuras reais a um ARX simulado sem essas medições.", ""]
        (out / "resumo.md").write_text("\n".join(text))
        print(f"Experimento concluído: {out.resolve()}")
        print(f"Modelo selecionado: {model.name}")
        for row in rows:
            print(f"{row['model']:14s} H={row['horizon_steps']:2d} RMSE trajetória={row['rmse_trajectory_m']:.6e} m")
        return out
    except Exception as error:
        write_json(out / "execucao.json", {"status": "falhou", "error": str(error)})
        raise


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "experimentos/config-linear.json")
    parser.add_argument("--output", type=Path, default=root / "resultados/linear-v1")
    args = parser.parse_args()
    run(args.config, args.output)

