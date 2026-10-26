"""Diagnóstico posterior ao teste do E003/E004: sensibilidade das referências clássicas à grade e à condição inicial.

ATENÇÃO: foi feito depois de ver as métricas de teste do E004. Serve para entender os resultados, não para
escolher modelos; nenhuma linha daqui substitui as execuções duffing-ruido-v1 e mamba3-duffing-ruido-v1.
O critério de escolha que o protocolo usaria (validação, 50 passos) é registrado ao lado do teste.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experimento_duffing import clean_metrics, fit_narx, monomials
from experimento_linear import evaluate, fit_arx, write_json

ROOT = Path(__file__).resolve().parent.parent


def last_rmse(result):
    return result["rows"][-1]["rmse_trajectory_m"] if result["rows"] else None


def scores(model, data, settings, clean_test=None, y_test_history=None):
    validation = last_rmse(evaluate(model, data["validacao_u"], data["validacao_y"], settings))
    test = evaluate(model, data["teste_u"], data["teste_y"] if y_test_history is None else y_test_history, settings)
    if clean_test is not None and test["rows"]:
        test = clean_metrics(test, clean_test, settings)
    return validation, last_rmse(test)


def main(output):
    if output.exists():
        raise FileExistsError(f"A pasta {output} já existe.")
    output.mkdir(parents=True)
    results = ROOT / "resultados"
    e3 = dict(np.load(results / "duffing-v1" / "dados.npz"))
    e4 = dict(np.load(results / "duffing-ruido-v1" / "dados.npz"))
    clean = dict(np.load(results / "duffing-ruido-v1" / "diagnostico" / "saida-sem-ruido.npz"))
    settings = json.loads((results / "duffing-ruido-v1" / "config.json").read_text())["evaluation"]
    report = {"aviso": __doc__.strip(), "metrica": "RMSE da trajetória de 50 passos [m]; no E004, teste contra a posição sem ruído",
              "narx_grau3_e004": [], "arx_e004": [], "arx_e003": [], "narx_grau3_e003": []}

    for order in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14):
        start = perf_counter()
        model = fit_narx(e4["treino_u"], e4["treino_y"], order, 3)
        validation, test = scores(model, e4, settings, clean["teste_y"])
        report["narx_grau3_e004"].append({"model": model.name, "parameters": model.parameters, "rank": model.diagnostics["rank"],
                                          "validation": validation, "test_clean": test, "seconds": perf_counter() - start})
        print(report["narx_grau3_e004"][-1], flush=True)
    for order in (2, 3, 4, 5, 6):
        model = fit_narx(e3["treino_u"], e3["treino_y"], order, 3)
        validation, test = scores(model, e3, settings)
        report["narx_grau3_e003"].append({"model": model.name, "parameters": model.parameters, "rank": model.diagnostics["rank"],
                                          "condition_number_scaled": model.diagnostics["condition_number_scaled"],
                                          "validation": validation, "test": test})
    for order in (8, 16, 32, 64):
        for key, data, target in (("arx_e004", e4, clean["teste_y"]), ("arx_e003", e3, None)):
            model = fit_arx(data["treino_u"], data["treino_y"], order, order)
            validation, test = scores(model, data, settings, target)
            report[key].append({"model": model.name, "validation": validation, "test": test})

    # De onde vem o erro do NARX(4,4) grau 3 no E004: coeficientes enviesados ou condição inicial ruidosa?
    noisy_fit = fit_narx(e4["treino_u"], e4["treino_y"], 4, 3)
    clean_fit = fit_narx(e4["treino_u"], clean["treino_y"], 4, 3)
    report["decomposicao_narx44"] = {
        "ajuste_ruidoso_condicao_inicial_ruidosa": scores(noisy_fit, e4, settings, clean["teste_y"])[1],
        "ajuste_ruidoso_condicao_inicial_limpa": scores(noisy_fit, e4, settings, clean["teste_y"], clean["teste_y"])[1],
        "ajuste_limpo_condicao_inicial_ruidosa": scores(clean_fit, e4, settings, clean["teste_y"])[1]}

    # Posto dos monômios só de entrada: patamares de no mínimo 5 amostras anulam produtos de diferenças.
    u = e3["treino_u"]
    t = np.arange(3, u.shape[1])
    z = np.stack([u[:, t - lag] for lag in range(4)] + [np.ones((len(u), len(t)))], axis=-1).reshape(-1, 5)
    design = np.prod(z[:, monomials(4, 3)], axis=2)
    report["posto_monomios_so_entrada_n4_grau3"] = {"rank": int(np.linalg.matrix_rank(design)), "columns": design.shape[1]}
    write_json(output / "diagnostico.json", report)
    print(json.dumps({k: v for k, v in report.items() if k.startswith(("decomposicao", "posto"))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "resultados" / "diagnostico-e004-v1")
    main(parser.parse_args().output)
