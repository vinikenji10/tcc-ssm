"""Identificação com Mamba-3: treino, escolha na validação e teste nas mesmas origens das referências clássicas.

Usa os dados de uma execução de referência (E001, planta linear; E003, Duffing).
Convenção do E001: no instante t a rede recebe [u[t], y[t]] e prevê y[t+1].
O treino usa erro de um passo em janelas com estado reiniciado. Configuração e checkpoint são
escolhidos pela simulação de 50 passos na validação; o teste só é avaliado depois dessas escolhas.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn

from experimento_linear import (ARX, Persistence, evaluate, evaluation_origins, rollout, summarize_rollouts,
                                write_csv, write_json)
from experimento_duffing import PolynomialNARX, clean_metrics
from mamba3_referencia import Mamba3SISO

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = "state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa"


class Mamba3Regressor(nn.Module):
    """[u, y] normalizados → projeção → blocos Mamba-3 com norma e residual → y do instante seguinte."""

    def __init__(self, d_model=64, n_layers=1, **block):
        super().__init__()
        self.encoder = nn.Linear(2, d_model)
        self.norms = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(n_layers))
        self.blocks = nn.ModuleList(Mamba3SISO(d_model, **block) for _ in range(n_layers))
        self.final_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _run(self, x, states, sequence):
        h = self.encoder(x)
        new_states = []
        for norm, block, state in zip(self.norms, self.blocks, states or [None] * len(self.blocks)):
            out, state = block(norm(h), state=state, return_state=True) if sequence else block.step(norm(h), state)
            h = h + out
            new_states.append(state)
        return self.head(self.final_norm(h)).squeeze(-1), new_states

    def forward(self, x, states=None, return_states=False):
        """x: (batch, L, 2). Sequência inteira, com fwd_ref; usado no treino e no aquecimento."""
        y, states = self._run(x, states, sequence=True)
        return (y, states) if return_states else y

    def step(self, x, states):
        """Continua a partir dos estados, uma amostra por vez, com step_ref."""
        return self._run(x, states, sequence=False)


class Normalizer:
    """Estatísticas do treino da execução de referência, as mesmas para a entrada y e para o alvo."""

    def __init__(self, stats):
        self.u_mean, self.u_std = stats["u"]["mean"], stats["u"]["std"]
        self.y_mean, self.y_std = stats["y"]["mean"], stats["y"]["std"]

    def u(self, values):
        return (values - self.u_mean) / self.u_std

    def y(self, values):
        return (values - self.y_mean) / self.y_std

    def y_physical(self, values):
        return values * self.y_std + self.y_mean

    def pairs(self, u, y):
        return np.stack([self.u(u), self.y(y)], axis=-1)


def training_windows(u, y, norm, window, stride):
    """Entrada [u[t], y[t]] e alvo y[t+1], em janelas que nunca cruzam trajetórias."""
    inputs, targets = norm.pairs(u, y[:, :-1]), norm.y(y[:, 1:])
    starts = range(0, u.shape[1] - window + 1, stride)
    x = np.stack([inputs[e, s:s + window] for e in range(len(u)) for s in starts])
    target = np.stack([targets[e, s:s + window] for e in range(len(u)) for s in starts])
    return x, target


def rollout_pairs(n_episodes, origins):
    return [(e, int(t)) for e in range(n_episodes) for t in origins]  # mesma ordem de `evaluate`


def simulate(model, u, y, origins, history, horizon, norm, device):
    """Previsões (trajetórias × origens, horizonte) em metros.

    Em cada origem t: aquece com [u, y] medidos em t-history+1..t, prevê y[t+1] e realimenta as
    previsões até y[t+horizon]. Só y[:t+1] é lido; as entradas futuras u[t+1:] são conhecidas, como no ARX.
    """
    pairs = rollout_pairs(len(u), origins)
    if min(t for _, t in pairs) + 1 < history:
        raise ValueError("Histórico maior que as medições disponíveis antes da origem.")
    measured = np.stack([norm.pairs(u[e, t - history + 1:t + 1], y[e, t - history + 1:t + 1]) for e, t in pairs])
    future_u = np.stack([norm.u(u[e, t + 1:t + horizon]) for e, t in pairs])
    measured = torch.tensor(measured, dtype=torch.float32, device=device)
    future_u = torch.tensor(future_u, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        out, states = model(measured, return_states=True)
        y_hat = out[:, -1]
        predictions = [y_hat]
        for j in range(horizon - 1):
            out, states = model.step(torch.stack([future_u[:, j], y_hat], dim=-1).unsqueeze(1), states)
            y_hat = out[:, -1]
            predictions.append(y_hat)
    return norm.y_physical(torch.stack(predictions, dim=1).cpu().double().numpy())


def rollout_targets(y, origins, horizon):
    return np.stack([y[e, t + 1:t + horizon + 1] for e, t in rollout_pairs(len(y), origins)])


def evaluate_network(model, name, u, y, settings, history, norm, device):
    origins = evaluation_origins(u.shape[1], settings)
    hmax = max(settings["horizons"])
    predictions = simulate(model, u, y, origins, history, hmax, norm, device)
    parameters = sum(p.numel() for p in model.parameters())
    return summarize_rollouts(name, parameters, predictions, rollout_targets(y, origins, hmax), settings, len(origins))


def merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        merged[key] = merge(merged[key], value) if isinstance(value, dict) else value
    return merged


def train_run(network, training, seed, signals, norm, settings, history, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Mamba3Regressor(**network).to(device)
    x, target = training_windows(signals["treino_u"], signals["treino_y"], norm,
                                 training["window"], training["window_stride"])
    x = torch.tensor(x, dtype=torch.float32, device=device)
    target = torch.tensor(target, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    batches = math.ceil(len(x) / training["batch_size"])
    total_updates = training["max_epochs"] * batches
    schedule = (lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total_updates) / total_updates))) \
        if training["lr_schedule"] == "cosine" else (lambda k: 1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    skip = training["loss_skip"]

    epochs, evaluations, best = [], [], {"score": math.inf, "epoch": None, "state": None}
    status, start = "concluido", perf_counter()
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        losses = []
        for index in np.array_split(rng.permutation(len(x)), batches):
            index = torch.as_tensor(index, device=device)
            loss = (model(x[index])[:, skip:] - target[index][:, skip:]).square().mean()
            if not torch.isfinite(loss):
                status = "perda não finita"
                break
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        if status != "concluido":
            break
        epochs.append({"epoch": epoch, "train_loss": statistics.fmean(losses)})
        out_of_time = (perf_counter() - start) / 60 > training["max_minutes_per_run"]
        if epoch % training["eval_every"] == 0 or epoch == training["max_epochs"] or out_of_time:
            result = evaluate_network(model, "validação", signals["validacao_u"], signals["validacao_y"],
                                      settings, history, norm, device)
            score = result["rows"][-1]["rmse_trajectory_m"] if result["rows"] else None
            evaluations.append({"epoch": epoch, "validation_rmse_trajectory_m": score,
                                "nonfinite_values": result["nonfinite_values"],
                                "elapsed_s": perf_counter() - start})
            if score is not None and score < best["score"]:
                best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if out_of_time:
            status = "limite de tempo"
            break
    return {"seed": seed, "status": status, "best_epoch": best["epoch"],
            "best_validation_rmse_trajectory_m": best["score"] if best["state"] else None,
            "train_seconds": perf_counter() - start, "windows": len(x), "updates_per_epoch": batches,
            "parameters": sum(p.numel() for p in model.parameters()),
            "epochs": epochs, "evaluations": evaluations}, best["state"]


def load_arx(path):
    spec = json.loads(Path(path).read_text())
    return ARX(spec["output_order"], spec["input_order"], spec["coefficients"], spec.get("diagnostics"))


def load_references(data_dir):
    """Persistência, ARX e, se a execução de referência tiver, NARX polinomial."""
    references = [Persistence(), load_arx(data_dir / "modelo-arx.json")]
    if (data_dir / "modelo-narx.json").exists():
        references.append(PolynomialNARX.from_dict(json.loads((data_dir / "modelo-narx.json").read_text())))
    return references


def check_against_reference(rows, data_dir):
    """As métricas das referências clássicas recalculadas aqui devem ser as gravadas na execução de referência."""
    with (data_dir / "metricas-teste.csv").open() as f:
        stored = {(r["model"], int(r["horizon_steps"])): float(r["rmse_trajectory_m"]) for r in csv.DictReader(f)}
    for row in rows:
        reference = stored[(row["model"], row["horizon_steps"])]
        if not math.isclose(row["rmse_trajectory_m"], reference, rel_tol=1e-9, abs_tol=1e-18):
            raise RuntimeError(f"{row['model']} H={row['horizon_steps']} difere da execução de referência.")


def make_plots(out, signals, references, nets, results, runs, settings, history, norm, device, sample_time):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "figure.dpi": 120,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    colors = {"real": "#112D4E", "arx": "#3F72AF", "narx": "#B91C1C", "persistence": "#D97706", "mamba": "#059669"}

    def color(name):
        return (colors["persistence"] if name == "Persistência" else colors["narx"] if name.startswith("NARX")
                else colors["arx"] if name.startswith("ARX") else colors["mamba"])
    figures = out / "figuras"
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    styles = ["-", "--", ":", "-."]
    palette = ["#059669", "#7C3AED", "#DC2626", "#0891B2"]
    for c, (candidate, candidate_runs) in enumerate(runs.items()):
        for s, run in enumerate(candidate_runs):
            label = f"{candidate}, semente {run['seed']}"
            ax[0].plot([e["epoch"] for e in run["epochs"]], [e["train_loss"] for e in run["epochs"]],
                       styles[s % 4], color=palette[c % 4], lw=1, label=label)
            ax[1].plot([e["epoch"] for e in run["evaluations"]],
                       [e["validation_rmse_trajectory_m"] for e in run["evaluations"]],
                       styles[s % 4], color=palette[c % 4], lw=1, marker=".", label=label)
    ax[0].set(yscale="log", xlabel="Época", ylabel="Erro quadrático médio (normalizado)",
              title="Treino: erro de um passo")
    ax[1].set(yscale="log", xlabel="Época", ylabel="RMSE da trajetória [m]",
              title="Validação: simulação de 50 passos")
    ax[1].legend(fontsize=7)
    save(fig, "01-curvas-treino")

    u, y = signals["teste_u"][:1], signals["teste_y"][:1]
    origin, n = settings["warmup"], 250
    first = next(iter(nets))
    one_step = simulate(nets[first], u, y, np.arange(origin, origin + n), history, 1, norm, device)[:, 0]
    actual = y[0, origin + 1:origin + n + 1]
    classical = [model for model in references if not isinstance(model, Persistence)]
    steps_classical = {model.name: np.array([model.predict_step(u[0, :t + 1], y[0, :t + 1]) for t in range(origin, origin + n)])
                       for model in classical}
    time = (np.arange(n) + origin + 1) * sample_time
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    ax[0].plot(time, actual, color=colors["real"], label="Medida")
    for name, values in steps_classical.items():
        ax[0].plot(time, values, "--", color=color(name), label=name)
    ax[0].plot(time, one_step, ":", color=colors["mamba"], lw=2, label=first)
    ax[0].set(ylabel="Posição [m]", title="Previsão de um passo com histórico medido real")
    ax[0].legend(ncols=4, fontsize=8)
    ax[1].plot(time, one_step - actual, color=colors["mamba"], label=f"Erro {first}")
    for name, values in steps_classical.items():
        ax[1].plot(time, values - actual, color=color(name), label=f"Erro {name}")
    ax[1].set(xlabel="Tempo [s]", ylabel="Previsto − medido [m]")
    ax[1].legend()
    save(fig, "02-previsao-um-passo")

    horizon = max(settings["horizons"])
    steps = np.arange(horizon + 1) * sample_time
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(steps, y[0, origin:origin + horizon + 1], color=colors["real"], lw=2, label="Medida")
    for model in references:
        values = np.r_[y[0, origin], rollout(model, u[0], y[0, :origin + 1], origin, horizon)]
        ax.plot(steps, values, "--", color=color(model.name), label=model.name)
    for name, net in nets.items():
        values = np.r_[y[0, origin], simulate(net, u, y, [origin], history, horizon, norm, device)[0]]
        ax.plot(steps, values, ":", color=colors["mamba"], lw=1.5, alpha=0.9, label=name)
    ax.axvline(0, color="#6B7280", lw=1)
    ax.set(xlabel="Tempo após a origem [s]", ylabel="Posição [m]",
           title=f"Simulação de {horizon} passos: saídas futuras reais não entram nos modelos")
    ax.legend(fontsize=8)
    save(fig, "03-simulacao-multiplos-passos")

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for name, result in results.items():
        if result["rmse_curve"] is None:
            continue
        ax.plot(np.arange(1, horizon + 1), result["rmse_curve"], color=color(name),
                alpha=1.0 if not name.startswith("Mamba") else 0.8, label=name)
    ax.set(yscale="log", xlabel="Passo futuro", ylabel="RMSE no ponto final [m]", title="Erro por horizonte no teste")
    ax.legend(fontsize=8)
    save(fig, "04-erro-por-horizonte")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config_path, output_path, device, smoke=False, validation_only=False):
    config_path, out = Path(config_path), Path(output_path)
    if out.exists():
        raise FileExistsError(f"A pasta {out} já existe. Escolha outra saída para preservar a execução.")
    config = json.loads(config_path.read_text())
    if smoke:  # só verifica se o fluxo completo roda; não serve como resultado
        config["training"].update(max_epochs=2, eval_every=1)
        config["candidates"], config["seeds"] = config["candidates"][:1], config["seeds"][:1]
    data_dir = ROOT / config["data"]
    signals = dict(np.load(data_dir / "dados.npz"))
    settings = json.loads((data_dir / "config.json").read_text())["evaluation"]
    stats = json.loads((data_dir / "normalizacao-treino.json").read_text())
    for name in ("u", "y"):
        if not math.isclose(stats[name]["std"], float(signals[f"treino_{name}"].std()), rel_tol=1e-12):
            raise RuntimeError("Normalização gravada não corresponde ao treino.")
    norm, history = Normalizer(stats), config["history"]

    out.mkdir(parents=True)
    write_json(out / "config.json", {**config, "device": device, "smoke": smoke})
    models_dir = out / "modelos"
    models_dir.mkdir()
    started = perf_counter()

    runs, states = {}, {}
    for candidate in config["candidates"]:
        network = merge(config["network"], candidate.get("network", {}))
        training = merge(config["training"], candidate.get("training", {}))
        runs[candidate["name"]] = []
        for seed in config["seeds"]:
            print(f"Treinando {candidate['name']}, semente {seed}...", flush=True)
            record, state = train_run(network, training, seed, signals, norm, settings, history, device)
            record.update(network=network, training=training)
            runs[candidate["name"]].append(record)
            if state is not None:
                path = models_dir / f"{candidate['name']}-semente{seed}.pt"
                torch.save(state, path)
                states[(candidate["name"], seed)] = (network, path)
            print(f"  {record['status']}; melhor validação {record['best_validation_rmse_trajectory_m']} m "
                  f"na época {record['best_epoch']}; {record['train_seconds']:.0f} s", flush=True)

    def candidate_score(candidate_runs):
        scores = [r["best_validation_rmse_trajectory_m"] for r in candidate_runs]
        return statistics.fmean(scores) if all(s is not None for s in scores) else math.inf

    scores = {name: candidate_score(candidate_runs) for name, candidate_runs in runs.items()}
    selected = min(scores, key=scores.get)
    if not math.isfinite(scores[selected]):
        raise RuntimeError("Nenhuma configuração completou todas as sementes com validação finita.")
    write_json(out / "selecao-validacao.json", {
        "criterion": config["selection"], "selected": selected, "scores_m": scores,
        "runs": runs})
    print(f"Selecionada pela validação: {selected}", flush=True)
    if validation_only:  # ajuste de treino durante o desenvolvimento, sem ler o teste
        return out

    # Teste: lido só agora, depois das escolhas.
    test_u, test_y = signals["teste_u"], signals["teste_y"]
    references = load_references(data_dir)
    arx = references[1]
    results = {model.name: evaluate(model, test_u, test_y, settings) for model in references}
    check_against_reference([row for r in results.values() for row in r["rows"]], data_dir)
    nets = {}
    for seed in config["seeds"]:
        network, path = states[(selected, seed)]
        net = Mamba3Regressor(**network).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        name = f"Mamba-3 (semente {seed})"
        results[name] = evaluate_network(net, name, test_u, test_y, settings, history, norm, device)
        if not np.array_equal(results[name]["target"], results[arx.name]["target"]):
            raise RuntimeError("Origens ou alvos diferentes dos usados pelo ARX.")
        nets[name] = net

    rows = [row for result in results.values() for row in result["rows"]]
    write_csv(out / "metricas-teste.csv", rows)
    clean_path = data_dir / "diagnostico" / "saida-sem-ruido.npz"
    clean_rows = []
    if clean_path.exists():  # dados com ruído: relatar também o erro em relação à posição sem ruído
        y_clean = np.load(clean_path)["teste_y"]
        clean_rows = [row for r in results.values() for row in clean_metrics(r, y_clean, settings)["rows"]]
        write_csv(out / "metricas-teste-sem-ruido.csv", clean_rows)
    write_json(out / "avaliacao-teste.json", {name: {k: v for k, v in r.items() if k not in ("prediction", "target")}
                                            for name, r in results.items()})
    summary = []
    for horizon in settings["horizons"]:
        values = [row["rmse_trajectory_m"] for row in rows if row["model"].startswith("Mamba") and row["horizon_steps"] == horizon]
        summary.append({"horizon_steps": horizon, "seeds": len(values), "mean_rmse_trajectory_m": statistics.fmean(values),
                        "std_rmse_trajectory_m": statistics.stdev(values) if len(values) > 1 else None,
                        "min_rmse_trajectory_m": min(values), "max_rmse_trajectory_m": max(values)})
    write_json(out / "resumo-sementes.json", summary)

    plant = json.loads((data_dir / "config.json").read_text())["plant"]
    make_plots(out, signals, references, nets, results, runs, settings, history, norm, device, plant["sample_time"])

    selected_runs = runs[selected]
    metadata = {"status": "concluido", "smoke": smoke,
                "created_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
                "scope": f"Mamba-3 SISO em PyTorch (funções de referência dos autores), {config.get('plant_description', 'planta linear do E001')}; MPC não executado",
                "upstream": UPSTREAM, "device": device, "precision": "float32",
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "numpy": np.__version__,
                "configuration_sha256": sha256(config_path),
                "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                  ("experimento_mamba3.py", "mamba3_referencia.py", "experimento_linear.py")},
                "data_sha256": sha256(data_dir / "dados.npz"),
                "total_seconds": perf_counter() - started,
                "selected": selected,
                "selected_train_seconds": [r["train_seconds"] for r in selected_runs],
                "hardware_regime": f"{device}; implementação de referência, sem benchmark de inferência"}
    write_json(out / "execucao.json", metadata)

    label = config.get("label", "E001")
    text = [f"# Mamba-3 nos dados do {label}", "",
            ("**Execução de verificação do fluxo (smoke): não usar como resultado.**" if smoke else
             f"Execução local concluída. {config.get('plant_description', 'Planta linear simulada, sem ruído')}; "
             f"mesmos dados, normalização, origens e métricas do {label}."), "",
            f"Implementação: Mamba-3 SISO em PyTorch, a partir das funções de referência de `{UPSTREAM}`, float32, dispositivo `{device}`.",
            f"Rede: Linear(2→{config['network']['d_model']}), {config['network']['n_layers']} bloco(s) Mamba-3 com RMSNorm e residual, "
            f"Linear({config['network']['d_model']}→1); {selected_runs[0]['parameters']} parâmetros.", "",
            "## Escolha na validação", "", f"Critério: {config['selection']}. Selecionada: **{selected}**.", "",
            "| Configuração | Semente | Melhor época | RMSE validação, 50 passos [m] | Tempo de treino [s] | Situação |",
            "|---|---:|---:|---:|---:|---|"]
    for name, candidate_runs in runs.items():
        for r in candidate_runs:
            score = r["best_validation_rmse_trajectory_m"]
            text.append(f"| {name} | {r['seed']} | {r['best_epoch']} | {score:.3e} | {r['train_seconds']:.0f} | {r['status']} |"
                        if score is not None else f"| {name} | {r['seed']} | - | - | {r['train_seconds']:.0f} | {r['status']} |")
    text += ["", "## Teste", "", "| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['model']} | {row['horizon_steps']} | {row['rmse_endpoint_m']:.3e} | "
                    f"{row['rmse_trajectory_m']:.3e} | {row['nrmse_trajectory']:.3e} |")
    text += ["", "Mamba-3, média entre sementes:", "", "| Horizonte | Média do RMSE da trajetória [m] | Desvio-padrão [m] | Mínimo [m] | Máximo [m] |",
             "|---:|---:|---:|---:|---:|"]
    for s in summary:
        std = f"{s['std_rmse_trajectory_m']:.2e}" if s["std_rmse_trajectory_m"] is not None else "-"
        text.append(f"| {s['horizon_steps']} | {s['mean_rmse_trajectory_m']:.3e} | {std} | {s['min_rmse_trajectory_m']:.3e} | {s['max_rmse_trajectory_m']:.3e} |")
    if clean_rows:
        text += ["", "Mesmas previsões, comparadas com a posição sem ruído (`metricas-teste-sem-ruido.csv`):", "",
                 "| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] |", "|---|---:|---:|---:|"]
        for row in clean_rows:
            text.append(f"| {row['model']} | {row['horizon_steps']} | {row['rmse_endpoint_m']:.3e} | {row['rmse_trajectory_m']:.3e} |")
    text += ["", "## Como ler", "",
             f"As referências clássicas foram recalculadas a partir dos modelos gravados no {label} e conferidas contra as métricas daquela execução. As previsões da rede usam as mesmas origens e os mesmos alvos.",
             f"Em cada origem, a rede é aquecida com as {history} medições até a origem, inclusive, e depois realimenta as próprias previsões; saídas futuras reais não entram.",
             *([config["reading_note"]] if "reading_note" in config else
               ["Na planta linear sem ruído, o ARX chega à precisão numérica. A pergunta aqui é se a rede aprende a dinâmica e quanto erra, não se supera o ARX."]),
             "Com três sementes, a dispersão é indicativa; não sustenta uma conclusão estatística ampla.",
             "Tempos de treino desta implementação de referência; não representam os kernels dos autores.", "",
             "## Artefatos", "", "- `config.json`, `selecao-validacao.json`: configuração, curvas e escolha.",
             "- `modelos/`: pesos do melhor checkpoint de cada execução.",
             "- `metricas-teste.csv`, `avaliacao-teste.json`, `resumo-sementes.json`: resultados no teste.",
             "- `figuras/`: curvas de treino, previsão de um passo, simulação de 50 passos e erro por horizonte.",
             "- `execucao.json`: versões, hashes e dispositivo.", ""]
    (out / "resumo.md").write_text("\n".join(text))
    print("\n".join(text[text.index("## Teste"):text.index("## Como ler")]))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experimentos/config-mamba3.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="duas épocas, uma semente: só testa o fluxo")
    parser.add_argument("--validation-only", action="store_true", help="para depois da escolha, sem ler o teste")
    args = parser.parse_args()
    run(args.config, args.output, args.device, args.smoke, args.validation_only)
