"""Invólucro de medição do G0: roda experimento_mamba3.run sem alterar o cálculo e grava tempos e memória.

Uso: python medir_treino.py <raiz do repo> <config> <saída> <dispositivo> <threads>
Grava <saída>/medicao-pc.json com tempo total, tempo gasto nas avaliações de validação,
tempo de treino estimado (total menos avaliações), épocas e pico de memória CUDA do PyTorch.
"""
import json
import os
import sys
from pathlib import Path
from time import perf_counter

root, config, output, device, threads = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.environ.setdefault("OMP_NUM_THREADS", str(threads))
sys.path.insert(0, str(Path(root) / "experimentos"))

import torch  # noqa: E402

torch.set_num_threads(threads)
import experimento_mamba3 as em  # noqa: E402

eval_times = []
original = em.evaluate_network


def timed_evaluate(*args, **kwargs):
    if device == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    result = original(*args, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    eval_times.append(perf_counter() - start)
    return result


em.evaluate_network = timed_evaluate  # só cronometra; mesma função, mesmos argumentos
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
start = perf_counter()
out = em.run(config, output, device, smoke=False, validation_only=True)
total = perf_counter() - start
selection = json.loads((Path(output) / "selecao-validacao.json").read_text())
record = selection["runs"][next(iter(selection["runs"]))][0]
epochs = len(record["epochs"])
report = {
    "device": device, "threads": threads, "pid": os.getpid(), "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    "wall_seconds_run": total, "train_seconds_recorded": record["train_seconds"],
    "epochs": epochs, "validation_evaluations": len(eval_times), "validation_eval_seconds": sum(eval_times),
    "train_only_seconds_estimate": record["train_seconds"] - sum(eval_times),
    "seconds_per_epoch_with_eval": record["train_seconds"] / epochs,
    "seconds_per_epoch_train_only": (record["train_seconds"] - sum(eval_times)) / epochs,
    "best_epoch": record["best_epoch"], "best_validation_rmse_trajectory_m": record["best_validation_rmse_trajectory_m"],
    "cuda_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
    "cuda_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None,
}
(Path(output) / "medicao-pc.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, indent=2, ensure_ascii=False))
