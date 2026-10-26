# Referências clássicas na planta de Duffing

Execução local concluída. Oscilador de Duffing simulado (mola cúbica), sem ruído.

Não linearidade nas posições do treino: força cúbica / força linear = 1.08 (RMS), até 3.6 na maior posição (0.848 m).

Escolhidos pela validação: **ARX(8,8)** (17 parâmetros) e **NARX(4,4) grau 3** (165 parâmetros).

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

## Artefatos

- `dados.npz`, `normalizacao-treino.json`: sinais e estatísticas do treino.
- `selecao-validacao.json`, `modelo-arx.json`, `modelo-narx.json`: candidatos, escolha e coeficientes.
- `metricas-teste.csv`, `avaliacao-teste.json`: resultados no teste.
- `figuras/`: entrada e saída, não linearidade e erro por horizonte.
- `diagnostico/`: estados físicos e medida da não linearidade; fora dos modelos.
