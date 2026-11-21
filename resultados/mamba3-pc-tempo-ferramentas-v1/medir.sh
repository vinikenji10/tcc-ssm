#!/usr/bin/env bash
# Uso: medir.sh <rótulo> <comando...>. Amostra a GPU (1 s) e o WSL (20 s) enquanto o comando roda.
label=$1; shift
S="$SCRATCH"
mkdir -p "$S/med"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$S/med/$label-processos-antes.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 1 > "$S/med/$label-gpu.csv" &
G=$!
( while true; do echo "$(date +%H:%M:%S) $(wsl.exe -e bash -c 'ps -eo pid,etime,pcpu,args | grep -i [p]ython | grep -v unattended' 2>/dev/null | tr '\n' ';')" >> "$S/med/$label-wsl.txt"; sleep 20; done ) &
W=$!
start=$(date +%s%N)
"$@" > "$S/med/$label-saida.log" 2>&1
rc=$?
end=$(date +%s%N)
kill $G $W 2>/dev/null
echo "rc=$rc parede_s=$(( (end - start) / 1000000 ))e-3" | tee "$S/med/$label-fim.txt"
echo "mem_max_MiB=$(cut -d, -f3 "$S/med/$label-gpu.csv" | tr -dc '0-9\n' | sort -n | tail -1)"
echo "wsl_python_vistos=$(grep -c 'python' "$S/med/$label-wsl.txt")"
tail -25 "$S/med/$label-saida.log"
