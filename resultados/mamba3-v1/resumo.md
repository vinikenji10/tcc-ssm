# Mamba-3 nos dados do E001

Execução local concluída. Planta linear simulada, sem ruído; mesmos dados, normalização, origens e métricas do E001.

Implementação: Mamba-3 SISO em PyTorch, a partir das funções de referência de `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`, float32, dispositivo `mps`.
Rede: Linear(2→64), 1 bloco(s) Mamba-3 com RMSNorm e residual, Linear(64→1); 34949 parâmetros.

## Escolha na validação

Critério: média, entre as sementes, do menor RMSE da trajetória de 50 passos na validação. Selecionada: **lr3e-3**.

| Configuração | Semente | Melhor época | RMSE validação, 50 passos [m] | Tempo de treino [s] | Situação |
|---|---:|---:|---:|---:|---|
| lr1e-3 | 1 | 270 | 1.948e-02 | 222 | concluido |
| lr1e-3 | 2 | 300 | 1.556e-02 | 222 | concluido |
| lr1e-3 | 3 | 290 | 2.677e-02 | 233 | concluido |
| lr3e-3 | 1 | 270 | 1.301e-02 | 243 | concluido |
| lr3e-3 | 2 | 280 | 1.589e-02 | 235 | concluido |
| lr3e-3 | 3 | 280 | 1.623e-02 | 262 | concluido |
| lr1e-2 | 1 | 280 | 4.250e-02 | 264 | concluido |
| lr1e-2 | 2 | 270 | 1.823e-02 | 286 | concluido |
| lr1e-2 | 3 | 290 | 2.083e-02 | 279 | concluido |

## Teste

| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |
|---|---:|---:|---:|---:|
| Persistência | 1 | 4.195e-02 | 4.195e-02 | 8.838e-02 |
| Persistência | 10 | 4.022e-01 | 2.534e-01 | 5.339e-01 |
| Persistência | 50 | 6.582e-01 | 6.839e-01 | 1.441e+00 |
| ARX(2,2) | 1 | 8.832e-16 | 8.832e-16 | 1.861e-15 |
| ARX(2,2) | 10 | 4.024e-14 | 2.142e-14 | 4.513e-14 |
| ARX(2,2) | 50 | 1.754e-13 | 1.096e-13 | 2.310e-13 |
| Mamba-3 (semente 1) | 1 | 1.018e-03 | 1.018e-03 | 2.146e-03 |
| Mamba-3 (semente 1) | 10 | 8.544e-03 | 5.397e-03 | 1.137e-02 |
| Mamba-3 (semente 1) | 50 | 1.337e-02 | 1.049e-02 | 2.211e-02 |
| Mamba-3 (semente 2) | 1 | 1.244e-03 | 1.244e-03 | 2.621e-03 |
| Mamba-3 (semente 2) | 10 | 9.479e-03 | 5.996e-03 | 1.263e-02 |
| Mamba-3 (semente 2) | 50 | 1.688e-02 | 1.247e-02 | 2.628e-02 |
| Mamba-3 (semente 3) | 1 | 1.161e-03 | 1.161e-03 | 2.447e-03 |
| Mamba-3 (semente 3) | 10 | 9.647e-03 | 6.196e-03 | 1.305e-02 |
| Mamba-3 (semente 3) | 50 | 1.514e-02 | 1.193e-02 | 2.513e-02 |

Mamba-3, média entre sementes:

| Horizonte | Média do RMSE da trajetória [m] | Desvio-padrão [m] | Mínimo [m] | Máximo [m] |
|---:|---:|---:|---:|---:|
| 1 | 1.141e-03 | 1.14e-04 | 1.018e-03 | 1.244e-03 |
| 10 | 5.863e-03 | 4.16e-04 | 5.397e-03 | 6.196e-03 |
| 50 | 1.163e-02 | 1.02e-03 | 1.049e-02 | 1.247e-02 |

## Como ler

Persistência e ARX foram recalculados a partir do modelo gravado no E001 e conferidos contra as métricas daquela execução. As previsões da rede usam as mesmas 545 origens e os mesmos alvos.
Em cada origem, a rede é aquecida com as 256 medições até a origem, inclusive, e depois realimenta as próprias previsões; saídas futuras reais não entram.
Na planta linear sem ruído, o ARX chega à precisão numérica. A pergunta aqui é se a rede aprende a dinâmica e quanto erra, não se supera o ARX.
Com três sementes, a dispersão é indicativa; não sustenta uma conclusão estatística ampla.
Tempos de treino desta implementação de referência; não representam os kernels dos autores.

## Artefatos

- `config.json`, `selecao-validacao.json`: configuração, curvas e escolha.
- `modelos/`: pesos do melhor checkpoint de cada execução.
- `metricas-teste.csv`, `avaliacao-teste.json`, `resumo-sementes.json`: resultados no teste.
- `figuras/`: curvas de treino, previsão de um passo, simulação de 50 passos e erro por horizonte.
- `execucao.json`: versões, hashes e dispositivo.
