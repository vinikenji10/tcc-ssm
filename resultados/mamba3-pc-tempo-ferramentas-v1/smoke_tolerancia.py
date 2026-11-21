"""Diagnóstico do G0: smoke do E002 com a conferência das referências clássicas registrada e tolerância absoluta de 1e-12 m.

Só substitui check_against_reference (comparação de métricas de persistência e ARX com o E001 do Mac);
treino, validação e teste da rede seguem o código do repositório sem alteração.
Uso: python smoke_tolerancia.py <raiz do repo> <saída> <dispositivo>
"""
import csv
import json
import math
import sys
from pathlib import Path

root, output, device = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, str(Path(root) / "experimentos"))
import experimento_mamba3 as em  # noqa: E402

log = []


def check_with_log(rows, data_dir):
    with (data_dir / "metricas-teste.csv").open(encoding="utf-8") as f:
        stored = {(r["model"], int(r["horizon_steps"])): float(r["rmse_trajectory_m"]) for r in csv.DictReader(f)}
    for row in rows:
        reference = stored[(row["model"], row["horizon_steps"])]
        value = row["rmse_trajectory_m"]
        log.append({"model": row["model"], "horizon_steps": row["horizon_steps"], "pc": value, "mac": reference,
                    "abs_diff": abs(value - reference),
                    "original_check_passes": math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-18)})
        if not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"{row['model']} H={row['horizon_steps']} difere além de 1e-12 m.")


em.check_against_reference = check_with_log
em.run(em.ROOT / "experimentos/config-mamba3.json", output, device, smoke=True)
(Path(output) / "diagnostico-conferencia-referencias.json").write_text(
    json.dumps({"nota": __doc__, "comparacoes": log}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(log, indent=1, ensure_ascii=False))
