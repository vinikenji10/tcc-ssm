#!/usr/bin/env bash
# Amostra a GPU por 15 s; imprime máximo de uso e memória, e o processo python do WSL, se houver.
max=0; mem=0
for i in $(seq 1 15); do
  read u m < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')
  (( u > max )) && max=$u; mem=$m; sleep 1
done
wslpy=$(wsl.exe -e bash -c "ps -eo pid,etime,pcpu,args | grep -i [p]ython | grep -v unattended" 2>/dev/null | tr '\n' ';')
echo "$(date +%H:%M:%S) uso_max=${max}% mem=${mem}MiB wsl_python=[${wslpy}]"
