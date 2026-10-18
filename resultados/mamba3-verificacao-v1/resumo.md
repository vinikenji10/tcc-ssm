# Verificação do Mamba-3 SISO em PyTorch - 2026-10-17

Implementação: `experimentos/mamba3_referencia.py`, a partir de `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`.
Bloco: {'d_model': 64, 'd_state': 64, 'expand': 2, 'headdim': 64}; float32; torch 2.14.1; macOS-26.6.2-arm64-arm-64bit.

| Medida | cpu | mps |
|---|---:|---:|
| Erro relativo máx., step_ref contra fwd_ref | 3.0e-07 | 2.5e-06 |
| Erro relativo máx., aquecimento 256 + 256 passos | 1.3e-06 | 2.7e-06 |
| Saídas antes de t=299 idênticas após mudar u a partir de t=300 | sim | sim |
| Gradientes finitos em todos os parâmetros | sim | sim |
| Passo de treino, batch 32 × 512 [s, mediana] | 0.198 | 0.081 |
| Avaliação E001: 545 origens, 256 + 50 passos [s, mediana] | 0.65 | 0.47 |

Rede medida: Linear(2→64), um bloco, Linear(64→1); 34821 parâmetros. Sinais aleatórios, sem os dados do E001.
Tempos desta implementação de referência; não representam os kernels dos autores nem um benchmark de latência.
