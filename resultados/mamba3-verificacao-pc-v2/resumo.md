# Verificação do Mamba-3 SISO em PyTorch - 2026-11-21

Implementação: `experimentos/mamba3_referencia.py`, a partir de `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`.
Bloco: {'d_model': 64, 'd_state': 64, 'expand': 2, 'headdim': 64}; float32; torch 2.14.1+cu132; Windows-11-10.0.26300-SP0.

| Medida | cpu | cuda |
|---|---:|---:|
| Erro relativo máx., step_ref contra fwd_ref | 3.6e-07 | 3.9e-07 |
| Erro relativo máx., aquecimento 256 + 256 passos | 6.3e-07 | 6.5e-07 |
| Saídas antes de t=299 idênticas após mudar u a partir de t=300 | sim | sim |
| Gradientes finitos em todos os parâmetros | sim | sim |
| Passo de treino, batch 32 × 512 [s, mediana] | 0.438 | 0.101 |
| Avaliação E001: 545 origens, 256 + 50 passos [s, mediana] | 1.42 | 0.69 |

Rede medida: Linear(2→64), um bloco, Linear(64→1); 34821 parâmetros. Sinais aleatórios, sem os dados do E001.
Tempos desta implementação de referência; não representam os kernels dos autores nem um benchmark de latência.
