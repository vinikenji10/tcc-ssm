#!/usr/bin/env bash
cd /c/Users/Vinicius/Projetos/TCC_SSM/tcc-ssm
pre() { echo "== $1"; bash "$SCRATCH/gpu_livre.sh"; tasklist | grep -i python | grep -v "$$" ; }
export PYTHONUTF8=1
pre tempo300b; bash "$SCRATCH/medir.sh" tempo300b .venv/Scripts/python.exe "$SCRATCH/medir_treino.py" "$PWD" experimentos/config-mamba3-pc-tempo.json resultados/mamba3-pc-tempo-300ep-v2 cuda 6
pre verif; bash "$SCRATCH/medir.sh" verif .venv/Scripts/python.exe experimentos/verificar_mamba3.py --output resultados/mamba3-verificacao-pc-v2
for n in 1 3 5; do pre conc-p$n; bash "$SCRATCH/medir.sh" conc-p$n bash "$SCRATCH/concorrencia.sh" $n 2; done
pre cpu6; bash "$SCRATCH/medir.sh" cpu6 env OMP_NUM_THREADS=6 .venv/Scripts/python.exe "$SCRATCH/medir_treino.py" "$PWD" "$SCRATCH/cfg/cpu-10ep-s1.json" resultados/mamba3-pc-tempo-cpu-10ep-v1 cpu 6
echo FIM
