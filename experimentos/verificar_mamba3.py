"""Primeira execução registrada do Mamba-3 SISO em PyTorch: equivalência, causalidade, gradientes e tempos.

Não treina modelo nem usa os dados do E001; mede se o bloco está pronto para o experimento E002.
Os tempos valem para esta implementação de referência, não para os kernels dos autores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import einops
import torch
import torch.nn as nn

from mamba3_referencia import Mamba3SISO

UPSTREAM = "state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa"
CONFIG = {"d_model": 64, "d_state": 64, "expand": 2, "headdim": 64}
SEED = 20261017


def relative_max_error(value, reference):
    return float((value - reference).abs().max() / reference.abs().max())


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def timed(function, device, repeats):
    function()  # aquecimento, fora da medida
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        synchronize(device)
        times.append(perf_counter() - start)
    return {"median_s": statistics.median(times), "max_s": max(times), "repeats": repeats}


def checks(device):
    torch.manual_seed(SEED)
    block = Mamba3SISO(**CONFIG, device=device)
    u = torch.randn(3, 512, CONFIG["d_model"], device=device)
    with torch.no_grad():
        y_fwd, state_fwd = block(u, return_state=True)
        y_step, state_step = block.step(u)
        _, state = block(u[:, :256], return_state=True)
        continued = []
        for t in range(256, u.shape[1]):
            y_t, state = block.step(u[:, t:t + 1], state)
            continued.append(y_t)
        changed = u.clone()
        changed[:, 300:] = torch.randn_like(changed[:, 300:])
        y_changed = block(changed)
    x = u.clone().requires_grad_()
    block(x).square().mean().backward()
    grads = [p.grad for p in block.parameters()] + [x.grad]
    return {
        "step_ref_vs_fwd_ref_output": relative_max_error(y_step, y_fwd),
        "step_ref_vs_fwd_ref_states": dict(zip(("SSM", "K", "V"),
                                               (relative_max_error(a, b) for a, b in zip(state_step[1:], state_fwd[1:])))),
        "warm_start_256_then_step_256": relative_max_error(torch.cat(continued, 1), y_fwd[:, 256:]),
        "past_outputs_unchanged_before_t299": bool(torch.equal(y_changed[:, :299], y_fwd[:, :299])),
        "rounding_at_t299": relative_max_error(y_changed[:, 299], y_fwd[:, 299]),
        "gradients_finite": bool(all(torch.isfinite(g).all() for g in grads)),
        "parameters_without_gradient": [n for n, p in block.named_parameters() if p.grad.abs().max() == 0],
    }


def timings(device):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(2, CONFIG["d_model"]), Mamba3SISO(**CONFIG),
                          nn.Linear(CONFIG["d_model"], 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    signals = torch.randn(32, 512, 2, device=device)
    target = torch.randn(32, 512, 1, device=device)

    def train_step():
        loss = (model(signals) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Avaliação do E001: 545 origens, aquecimento com 256 medições e 50 passos realimentados.
    encoder, block, head = model
    history = torch.randn(545, 256, 2, device=device)
    future_u = torch.randn(545, 50, 1, device=device)

    @torch.no_grad()
    def rollout_batch():
        _, state = block(encoder(history), return_state=True)
        y_prev = history[:, -1:, 1:]
        for j in range(50):
            out, state = block.step(encoder(torch.cat([future_u[:, j:j + 1], y_prev], dim=-1)), state)
            y_prev = head(out)

    return {
        "training_step_batch32_len512": timed(train_step, device, repeats=5),
        "rollout_545_origins_warmup256_h50": timed(rollout_batch, device, repeats=3),
        "network_parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="pasta nova para os resultados")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        sys.exit(f"{out} já existe; use uma pasta nova para preservar execuções anteriores.")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    source = Path(__file__).with_name("mamba3_referencia.py")
    report = {
        "created_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "scope": "bloco Mamba-3 SISO em PyTorch puro; sem treino nos dados do E001",
        "upstream": UPSTREAM,
        "block_config": CONFIG,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "einops": einops.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precision": "float32",
        "devices": {device: {"checks": checks(device), "timings": timings(device)} for device in devices},
    }
    out.mkdir(parents=True)
    (out / "verificacao.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [f"# Verificação do Mamba-3 SISO em PyTorch - {report['created_at'][:10]}", "",
             f"Implementação: `experimentos/mamba3_referencia.py`, a partir de `{UPSTREAM}`.",
             f"Bloco: {CONFIG}; float32; torch {torch.__version__}; {report['platform']}.", "",
             "| Medida | " + " | ".join(devices) + " |", "|---|" + "---:|" * len(devices)]
    rows = [("Erro relativo máx., step_ref contra fwd_ref", lambda c, t: f"{c['step_ref_vs_fwd_ref_output']:.1e}"),
            ("Erro relativo máx., aquecimento 256 + 256 passos", lambda c, t: f"{c['warm_start_256_then_step_256']:.1e}"),
            ("Saídas antes de t=299 idênticas após mudar u a partir de t=300",
             lambda c, t: "sim" if c["past_outputs_unchanged_before_t299"] else "não"),
            ("Gradientes finitos em todos os parâmetros",
             lambda c, t: "sim" if c["gradients_finite"] and not c["parameters_without_gradient"] else "não"),
            ("Passo de treino, batch 32 × 512 [s, mediana]",
             lambda c, t: f"{t['training_step_batch32_len512']['median_s']:.3f}"),
            ("Avaliação E001: 545 origens, 256 + 50 passos [s, mediana]",
             lambda c, t: f"{t['rollout_545_origins_warmup256_h50']['median_s']:.2f}")]
    for label, cell in rows:
        values = [cell(report["devices"][d]["checks"], report["devices"][d]["timings"]) for d in devices]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", f"Rede medida: Linear(2→64), um bloco, Linear(64→1); "
              f"{report['devices']['cpu']['timings']['network_parameters']} parâmetros. Sinais aleatórios, sem os dados do E001.",
              "Tempos desta implementação de referência; não representam os kernels dos autores nem um benchmark de latência."]
    (out / "resumo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
