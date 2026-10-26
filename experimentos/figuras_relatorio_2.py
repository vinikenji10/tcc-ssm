"""Figuras de comparação do relatório 2 (E003 e E004), a partir dos modelos já gravados.

Não treina nem escolhe nada: recarrega ARX, NARX e os checkpoints da rede selecionada e refaz as
simulações no teste. No E004, o erro é medido contra a posição sem ruído (diagnóstico da execução).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experimento_duffing import PolynomialNARX, clean_metrics
from experimento_linear import Persistence, evaluate, rollout, write_json
from experimento_mamba3 import Mamba3Regressor, Normalizer, evaluate_network, load_arx, simulate

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"Medida": "#9CA3AF", "Real": "#112D4E", "Persistência": "#D97706", "ARX": "#3F72AF",
          "NARX": "#B91C1C", "Mamba-3": "#059669"}


def family(name):
    return next((key for key in ("ARX", "NARX", "Mamba-3") if name.startswith(key)), name)


def load_run(reference, network_run, device):
    data_dir, net_dir = ROOT / reference, ROOT / network_run
    signals = dict(np.load(data_dir / "dados.npz"))
    settings = json.loads((data_dir / "config.json").read_text())["evaluation"]
    norm = Normalizer(json.loads((data_dir / "normalizacao-treino.json").read_text()))
    config = json.loads((net_dir / "config.json").read_text())
    selected = json.loads((net_dir / "selecao-validacao.json").read_text())["selected"]
    network = {**config["network"], **next(c for c in config["candidates"] if c["name"] == selected).get("network", {})}
    nets = {}
    for seed in config["seeds"]:
        net = Mamba3Regressor(**network).to(device)
        net.load_state_dict(torch.load(net_dir / "modelos" / f"{selected}-semente{seed}.pt", map_location=device))
        nets[f"Mamba-3 (semente {seed})"] = net
    classical = [Persistence(), load_arx(data_dir / "modelo-arx.json"),
                 PolynomialNARX.from_dict(json.loads((data_dir / "modelo-narx.json").read_text()))]
    clean_path = data_dir / "diagnostico" / "saida-sem-ruido.npz"
    clean = np.load(clean_path)["teste_y"] if clean_path.exists() else None
    return signals, settings, norm, config["history"], classical, nets, clean


def curves(run, device):
    signals, settings, norm, history, classical, nets, clean = run
    u, y = signals["teste_u"], signals["teste_y"]
    results = {m.name: evaluate(m, u, y, settings) for m in classical}
    results |= {name: evaluate_network(net, name, u, y, settings, history, norm, device) for name, net in nets.items()}
    if clean is not None:
        results = {name: clean_metrics(r, clean, settings) for name, r in results.items()}
    return {name: np.array(r["rmse_curve"]) for name, r in results.items()}, \
        {name: r["rows"] for name, r in results.items()}


def main(output, device):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "resultados" / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
    output.mkdir(parents=True, exist_ok=True)
    runs = {"E003: Duffing sem ruído": load_run("resultados/duffing-v1", "resultados/mamba3-duffing-v1", device),
            "E004: Duffing com ruído (erro contra a posição sem ruído)":
                load_run("resultados/duffing-ruido-v1", "resultados/mamba3-duffing-ruido-v1", device)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, layout="constrained")
    table = {}
    for ax, (title, run) in zip(axes, runs.items()):
        curve, rows = curves(run, device)
        table[title] = rows
        steps = np.arange(1, len(next(iter(curve.values()))) + 1)
        mamba = np.array([c for name, c in curve.items() if family(name) == "Mamba-3"])
        for name, c in curve.items():
            if family(name) != "Mamba-3":
                ax.plot(steps, c, color=COLORS[family(name)], label=name)
        ax.plot(steps, mamba.mean(axis=0), color=COLORS["Mamba-3"], label="Mamba-3 (média de 3 sementes)")
        ax.fill_between(steps, mamba.min(axis=0), mamba.max(axis=0), color=COLORS["Mamba-3"], alpha=0.2, lw=0)
        ax.set(yscale="log", xlabel="Passo futuro", title=title)
    axes[0].set_ylabel("RMSE no ponto final [m]")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "erro-horizonte-duffing.png")
    plt.close(fig)
    write_json(output / "metricas-figuras.json", table)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, layout="constrained")
    for ax, (title, run) in zip(axes, runs.items()):
        signals, settings, norm, history, classical, nets, clean = run
        u, y = signals["teste_u"][:1], signals["teste_y"][:1]
        origin, horizon = settings["warmup"], max(settings["horizons"])
        time = np.arange(horizon + 1) * 0.05
        if clean is not None:
            ax.plot(time, y[0, origin:origin + horizon + 1], ".", ms=3, color=COLORS["Medida"], label="Medida (com ruído)")
            ax.plot(time, clean[0, origin:origin + horizon + 1], color=COLORS["Real"], lw=2, label="Posição sem ruído")
        else:
            ax.plot(time, y[0, origin:origin + horizon + 1], color=COLORS["Real"], lw=2, label="Medida")
        for model in classical[1:]:
            ax.plot(time, np.r_[y[0, origin], rollout(model, u[0], y[0, :origin + 1], origin, horizon)], "--",
                    color=COLORS[family(model.name)], label=model.name)
        name, net = next(iter(nets.items()))
        ax.plot(time, np.r_[y[0, origin], simulate(net, u, y, [origin], history, horizon, norm, device)[0]], ":",
                lw=2, color=COLORS["Mamba-3"], label=name)
        ax.set(xlabel="Tempo após a origem [s]", title=title.split(" (")[0])
    axes[0].set_ylabel("Posição [m]")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.savefig(output / "simulacao-duffing.png")
    plt.close(fig)
    print(f"Figuras em {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "relatorio-2" / "figuras")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    main(args.output, args.device)
