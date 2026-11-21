"""Avaliação única no teste final (Onda 5 do plano A3).

Lê a seleção congelada, constrói os modelos pelos adaptadores registrados, carrega o teste final uma vez
(o carregamento fica no registro de uso) e grava métricas e intervalos de bootstrap. Nada é ajustado nem
escolhido aqui: os modelos são construídos antes de abrir o teste, e um erro de adaptador não consome o uso.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from experimento_linear import write_csv
from protocolo_v2 import (ROOT, bootstrap_families, build_members, divergence_limit, evaluate_members,
                          evaluation_settings, h3_verdict, load_development, load_final_test, load_frozen_selection,
                          now_iso, selection_data_dir, sha256_file, sha256_text, write_json)


def comparison_pairs(selection):
    """Comparações pré-registradas primeiro (com "melhor_tradicional" resolvido), depois os demais pares."""
    names = sorted(selection["familias"])

    def resolve(name):
        return selection["melhor_tradicional"] if name == "melhor_tradicional" else name

    declared = [tuple(resolve(n) for n in pair) for pair in selection.get("comparacoes", [])]
    others = [pair for pair in combinations(names, 2) if pair not in declared and pair[::-1] not in declared]
    return declared, others


def strip_replicates(result):
    return {k: v for k, v in result.items() if k != "replicates"}


def run(selection_path, output_path, justification=None, batch_size=None, code_root=ROOT):
    selection_path, out = Path(selection_path), Path(output_path)
    if out.exists():
        raise FileExistsError(f"A pasta {out} já existe. Escolha outra saída para preservar a execução.")
    selection = load_frozen_selection(selection_path, code_root=code_root)
    data_dir = selection_data_dir(selection, selection_path)
    development, manifest = load_development(data_dir)
    config_bytes = (data_dir / "config.json").read_bytes()
    if sha256_text(config_bytes) != selection["dados"]["configuracao_sha256"]:
        raise ValueError("A configuração dos dados não corresponde à seleção congelada.")
    config = json.loads(config_bytes)
    settings = evaluation_settings(config)
    context = {"selection_dir": selection_path.parent, "development": development, "config": config}
    families = {name: build_members(entry, context) for name, entry in selection["familias"].items()}

    out.mkdir(parents=True)
    test, usage = load_final_test(data_dir, selection_path, caller="avaliar_teste_final.py:run",
                                  justification=justification, code_root=code_root)
    y_clean = test.get("teste_final_y_limpo")
    limit = divergence_limit(development, settings["divergence_factor"])
    evaluations = {name: evaluate_members(members, test["teste_final_u"], test["teste_final_y"], settings,
                                          y_clean=y_clean, divergence_limit_m=limit, batch_size=batch_size)
                   for name, members in families.items()}

    def rows_for(target):
        rows = []
        for name, evaluation in evaluations.items():
            for seed, member in zip((m.get("semente") for m in selection["familias"][name]["membros"]),
                                    evaluation["members"]):
                rows += [{"family": name, "seed": seed, **row} for row in member[target]["rows"]]
        return rows

    write_csv(out / "metricas-teste-final.csv", rows_for("measured"))
    if y_clean is not None:
        write_csv(out / "metricas-teste-final-sem-ruido.csv", rows_for("clean"))

    declared, others = comparison_pairs(selection)
    boot = config["bootstrap"]
    origins = next(iter(evaluations.values()))["origins_per_episode"]
    targets = {"medida": "squared_errors"}
    if y_clean is not None:
        targets["sem_ruido"] = "squared_errors_clean"
    intervals = {}
    for target, key in targets.items():
        errors = {name: np.stack([m[key] for m in ev["members"]]) for name, ev in evaluations.items()}
        for kind in ("trajectory", "endpoint"):
            for horizon in settings["horizons"]:
                result = bootstrap_families(errors, origins, horizon, kind, boot["resamples"], boot["seed"],
                                            boot["confidence"], pairs=declared + others)
                for c in result["comparisons"]:
                    c["pre_registrada"] = (c["a"], c["b"]) in declared
                intervals[f"{target}/{kind}/{horizon}"] = strip_replicates(result)

    h3 = None
    if selection.get("h3"):
        spec = selection["h3"]
        competitors = [selection["melhor_tradicional"] if c == "melhor_tradicional" else c
                       for c in spec["competidores"]]
        errors = {name: np.stack([m["squared_errors"] for m in ev["members"]]) for name, ev in evaluations.items()}
        result = bootstrap_families(errors, origins, settings["selection_horizon"], "trajectory", boot["resamples"],
                                    boot["seed"], spec["confianca"],
                                    pairs=[(spec["rede"], c) for c in competitors])
        h3 = {**h3_verdict(result["comparisons"], spec["rede"], competitors), "confidence": spec["confianca"],
              "comparisons": result["comparisons"], "metric": "RMSE da trajetória no horizonte de escolha, medida"}

    write_json(out / "intervalos-bootstrap.json", {"intervals": intervals, "h3": h3})
    write_json(out / "avaliacao-teste-final.json", {
        name: [{"name": m["name"], "parameters": m["parameters"], "simulation_seconds": m["simulation_seconds"],
                "divergence": m["divergence"],
                "rows": m["measured"]["rows"], "rows_clean": m.get("clean", {}).get("rows"),
                "rmse_curve": m["measured"]["rmse_curve"]} for m in ev["members"]]
        for name, ev in evaluations.items()})
    write_json(out / "uso-teste-final.json", usage)
    write_json(out / "execucao.json", {
        "status": "concluido", "created_at": now_iso(), "scenario": selection["cenario"],
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "selection_sha256": sha256_file(selection_path), "test_sha256": usage["teste_final_sha256"],
        "code_sha256": selection["codigo_sha256"], "test_use": usage["uso"],
        "origins_per_trajectory": origins, "trajectories": int(len(test["teste_final_u"]))})

    main = intervals[f"medida/trajectory/{settings['selection_horizon']}"]
    text = [f"# Teste final: {selection['cenario']}", "",
            f"Leitura {usage['uso']} do teste final, em {usage['data']}. Seleção congelada: `{selection_path.name}`.", "",
            f"RMSE da trajetória de {settings['selection_horizon']} passos contra a medição; intervalo de "
            f"{boot['confidence']:.0%} por bootstrap ({boot['resamples']} reamostragens de trajetórias e sementes).", "",
            "| Família | Membros | RMSE [m] | Intervalo [m] | Divergência |", "|---|---:|---:|---|---|"]
    for name, family in main["families"].items():
        interval = "indefinido" if family["interval"] is None else \
            f"{family['interval'][0]:.3e} a {family['interval'][1]:.3e}"
        diverged = sum(m["divergence"]["diverged"] for m in evaluations[name]["members"])
        text.append(f"| {name} | {len(family['members'])} | {family['point']:.3e} | {interval} | "
                    f"{diverged} membro(s) |")
    text += ["", "| a | b | a − b [m] | Intervalo [m] | Veredito |", "|---|---|---:|---|---|"]
    for c in main["comparisons"]:
        interval = "indefinido" if c["interval"] is None else f"{c['interval'][0]:.3e} a {c['interval'][1]:.3e}"
        mark = " (pré-registrada)" if c["pre_registrada"] else ""
        text.append(f"| {c['a']} | {c['b']} | {c['difference']:.3e} | {interval} | {c['verdict']}{mark} |")
    if h3:
        text += ["", f"H3 neste cenário (confiança {h3['confidence']:.4f}): "
                 f"{'sustentada' if h3['sustained'] else 'não sustentada'}; vereditos {h3['verdicts']}."]
    (out / "resumo.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selecao", type=Path, required=True, help="arquivo de seleção congelada")
    parser.add_argument("--saida", type=Path, required=True, help="pasta nova para os resultados")
    parser.add_argument("--justificativa", default=None, help="obrigatória a partir da segunda leitura")
    parser.add_argument("--lote", type=int, default=None, help="tamanho do lote de origens por chamada")
    args = parser.parse_args()
    print(f"Resultados em {run(args.selecao, args.saida, args.justificativa, args.lote).resolve()}")
