# Mamba-3 nos dados do E003

Execução local concluída. Oscilador de Duffing (mola cúbica) simulado, sem ruído; mesmos dados, normalização, origens e métricas do E003.

Implementação: Mamba-3 SISO em PyTorch, a partir das funções de referência de `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`, float32, dispositivo `mps`.
Rede: Linear(2→64), 1 bloco(s) Mamba-3 com RMSNorm e residual, Linear(64→1); 34949 parâmetros.

## Escolha na validação

Critério: média, entre as sementes, do menor RMSE da trajetória de 50 passos na validação. Selecionada: **lr3e-3**.

| Configuração | Semente | Melhor época | RMSE validação, 50 passos [m] | Tempo de treino [s] | Situação |
|---|---:|---:|---:|---:|---|
| lr1e-3 | 1 | 280 | 1.893e-02 | 214 | concluido |
| lr1e-3 | 2 | 280 | 2.980e-02 | 213 | concluido |
| lr1e-3 | 3 | 270 | 2.438e-02 | 211 | concluido |
| lr3e-3 | 1 | 280 | 1.246e-02 | 215 | concluido |
| lr3e-3 | 2 | 300 | 2.322e-02 | 218 | concluido |
| lr3e-3 | 3 | 280 | 1.233e-02 | 216 | concluido |
| lr1e-2 | 1 | 300 | 1.142e-02 | 218 | concluido |
| lr1e-2 | 2 | 270 | 1.975e-02 | 227 | concluido |
| lr1e-2 | 3 | 270 | 1.717e-02 | 239 | concluido |

## Teste

| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |
|---|---:|---:|---:|---:|
| Persistência | 1 | 3.465e-02 | 3.465e-02 | 1.228e-01 |
| Persistência | 10 | 3.193e-01 | 2.038e-01 | 7.224e-01 |
| Persistência | 50 | 3.118e-01 | 3.797e-01 | 1.346e+00 |
| ARX(8,8) | 1 | 1.111e-05 | 1.111e-05 | 3.938e-05 |
| ARX(8,8) | 10 | 1.944e-02 | 8.369e-03 | 2.966e-02 |
| ARX(8,8) | 50 | 1.251e-01 | 8.915e-02 | 3.160e-01 |
| NARX(4,4) grau 3 | 1 | 4.256e-10 | 4.256e-10 | 1.509e-09 |
| NARX(4,4) grau 3 | 10 | 3.266e-09 | 1.998e-09 | 7.084e-09 |
| NARX(4,4) grau 3 | 50 | 8.091e-09 | 5.530e-09 | 1.960e-08 |
| Mamba-3 (semente 1) | 1 | 7.380e-04 | 7.380e-04 | 2.616e-03 |
| Mamba-3 (semente 1) | 10 | 7.979e-03 | 5.346e-03 | 1.895e-02 |
| Mamba-3 (semente 1) | 50 | 1.771e-02 | 1.257e-02 | 4.455e-02 |
| Mamba-3 (semente 2) | 1 | 1.170e-03 | 1.170e-03 | 4.149e-03 |
| Mamba-3 (semente 2) | 10 | 1.289e-02 | 8.199e-03 | 2.906e-02 |
| Mamba-3 (semente 2) | 50 | 3.196e-02 | 2.219e-02 | 7.866e-02 |
| Mamba-3 (semente 3) | 1 | 7.531e-04 | 7.531e-04 | 2.669e-03 |
| Mamba-3 (semente 3) | 10 | 6.991e-03 | 4.658e-03 | 1.651e-02 |
| Mamba-3 (semente 3) | 50 | 1.521e-02 | 1.101e-02 | 3.903e-02 |

Mamba-3, média entre sementes:

| Horizonte | Média do RMSE da trajetória [m] | Desvio-padrão [m] | Mínimo [m] | Máximo [m] |
|---:|---:|---:|---:|---:|
| 1 | 8.872e-04 | 2.45e-04 | 7.380e-04 | 1.170e-03 |
| 10 | 6.067e-03 | 1.88e-03 | 4.658e-03 | 8.199e-03 |
| 50 | 1.526e-02 | 6.06e-03 | 1.101e-02 | 2.219e-02 |

## Como ler

As referências clássicas foram recalculadas a partir dos modelos gravados no E003 e conferidas contra as métricas daquela execução. As previsões da rede usam as mesmas origens e os mesmos alvos.
Em cada origem, a rede é aquecida com as 256 medições até a origem, inclusive, e depois realimenta as próprias previsões; saídas futuras reais não entram.
Na planta de Duffing sem ruído, o NARX polinomial de grau 3 contém a não linearidade verdadeira (cúbica) e chega perto da precisão numérica; o ARX linear não a representa. A pergunta é onde a rede fica entre as duas referências clássicas.
Com três sementes, a dispersão é indicativa; não sustenta uma conclusão estatística ampla.
Tempos de treino desta implementação de referência; não representam os kernels dos autores.

## Artefatos

- `config.json`, `selecao-validacao.json`: configuração, curvas e escolha.
- `modelos/`: pesos do melhor checkpoint de cada execução.
- `metricas-teste.csv`, `avaliacao-teste.json`, `resumo-sementes.json`: resultados no teste.
- `figuras/`: curvas de treino, previsão de um passo, simulação de 50 passos e erro por horizonte.
- `execucao.json`: versões, hashes e dispositivo.
