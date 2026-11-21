#!/usr/bin/env bash
# Uso: concorrencia.sh <N processos> <threads por processo>. Lança N treinos de 30 épocas ao mesmo tempo (sementes 1..N).
N=$1; T=$2
S="$SCRATCH"
cd /c/Users/Vinicius/Projetos/TCC_SSM/tcc-ssm
pids=()
for k in $(seq 1 $N); do
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONUTF8=1 .venv/Scripts/python.exe "$S/medir_treino.py" "$PWD" "$S/cfg/conc-30ep-s$k.json" \
    "resultados/mamba3-pc-tempo-30ep-p$N-s$k-v1" cuda $T > "$S/med/conc-p$N-s$k.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
exit $rc
