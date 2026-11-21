"""Compara duas execuções do E001: métricas, coeficientes do ARX, dados e seleção."""
import csv, json, sys
from pathlib import Path
import numpy as np

a, b = Path(sys.argv[1]), Path(sys.argv[2])

def diff(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.abs(x - y)
    rel = d / np.maximum(np.abs(y), np.finfo(float).tiny)
    return float(d.max()), float(rel.max())

def rows(p):
    with (p / "metricas-teste.csv").open() as f:
        return {(r["model"], r["horizon_steps"]): r for r in csv.DictReader(f)}

ra, rb = rows(a), rows(b)
assert ra.keys() == rb.keys()
cols = [c for c in next(iter(ra.values())) if c not in ("model", "horizon_steps", "split")]
numeric = []
for k in ra:
    for c in cols:
        try:
            numeric.append((k, c, float(ra[k][c]), float(rb[k][c])))
        except ValueError:
            assert ra[k][c] == rb[k][c], (k, c)
abs_m, rel_m = diff([n[2] for n in numeric], [n[3] for n in numeric])
print(f"metricas-teste.csv: {len(numeric)} valores; dif. abs. máx {abs_m:.3e}; rel. máx {rel_m:.3e}")
for k, c, x, y in numeric:
    if x != y:
        print("   diferente:", k, c, x, y)
ma, mb = json.loads((a / "modelo-arx.json").read_text()), json.loads((b / "modelo-arx.json").read_text())
print("ordens ARX:", ma["output_order"], ma["input_order"], "|", mb["output_order"], mb["input_order"])
print("coeficientes ARX: dif. abs. máx %.3e; rel. máx %.3e" % diff(ma["coefficients"], mb["coefficients"]))
print("coeficientes idênticos bit a bit:", ma["coefficients"] == mb["coefficients"])
da, db = np.load(a / "dados.npz"), np.load(b / "dados.npz")
print("dados.npz idênticos:", sorted(da.files) == sorted(db.files) and all(np.array_equal(da[k], db[k]) for k in da.files))
sa, sb = json.loads((a / "selecao-validacao.json").read_text()), json.loads((b / "selecao-validacao.json").read_text())
fa = json.dumps(sa, sort_keys=True); fb = json.dumps(sb, sort_keys=True)
print("selecao-validacao.json idêntico:", fa == fb)
def nums(o):
    if isinstance(o, dict):
        for k in sorted(o):
            if "second" in k or "time" in k: continue
            yield from nums(o[k])
    elif isinstance(o, list):
        for v in o: yield from nums(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)
na, nb = list(nums(sa)), list(nums(sb))
if len(na) == len(nb):
    print("seleção (sem tempos): %d valores; dif. abs. máx %.3e; rel. máx %.3e" % ((len(na),) + diff(na, nb)))
print("normalizacao-treino.json idêntico:", (a / "normalizacao-treino.json").read_text() == (b / "normalizacao-treino.json").read_text())
