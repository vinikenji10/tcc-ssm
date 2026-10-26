# Referências clássicas na planta de Duffing

Execução local concluída. Oscilador de Duffing simulado (mola cúbica), com ruído de medição conforme configuração.

Não linearidade nas posições do treino: força cúbica / força linear = 1.08 (RMS), até 3.6 na maior posição (0.848 m).

Escolhidos pela validação: **ARX(8,8)** (17 parâmetros) e **NARX(4,4) grau 3** (165 parâmetros).

| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |
|---|---:|---:|---:|---:|
| Persistência | 1 | 4.418e-02 | 4.418e-02 | 1.564e-01 |
| Persistência | 10 | 3.193e-01 | 2.049e-01 | 7.255e-01 |
| Persistência | 50 | 3.132e-01 | 3.805e-01 | 1.347e+00 |
| ARX(8,8) | 1 | 2.735e-02 | 2.735e-02 | 9.684e-02 |
| ARX(8,8) | 10 | 7.911e-02 | 5.523e-02 | 1.955e-01 |
| ARX(8,8) | 50 | 1.273e-01 | 1.040e-01 | 3.680e-01 |
| NARX(4,4) grau 3 | 1 | 3.150e-02 | 3.150e-02 | 1.115e-01 |
| NARX(4,4) grau 3 | 10 | 9.996e-02 | 6.951e-02 | 2.461e-01 |
| NARX(4,4) grau 3 | 50 | 1.844e-01 | 1.463e-01 | 5.179e-01 |

Tabela acima: erro em relação à posição medida, com ruído. Abaixo, as mesmas previsões comparadas com a posição sem ruído (`metricas-teste-sem-ruido.csv`):

| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] |
|---|---:|---:|---:|
| Persistência | 1 | 4.058e-02 | 4.058e-02 |
| Persistência | 10 | 3.192e-01 | 2.043e-01 |
| Persistência | 50 | 3.130e-01 | 3.803e-01 |
| ARX(8,8) | 1 | 1.844e-02 | 1.844e-02 |
| ARX(8,8) | 10 | 7.629e-02 | 5.144e-02 |
| ARX(8,8) | 50 | 1.267e-01 | 1.021e-01 |
| NARX(4,4) grau 3 | 1 | 2.461e-02 | 2.461e-02 |
| NARX(4,4) grau 3 | 10 | 9.801e-02 | 6.717e-02 |
| NARX(4,4) grau 3 | 50 | 1.834e-01 | 1.453e-01 |

## Artefatos

- `dados.npz`, `normalizacao-treino.json`: sinais e estatísticas do treino.
- `selecao-validacao.json`, `modelo-arx.json`, `modelo-narx.json`: candidatos, escolha e coeficientes.
- `metricas-teste.csv`, `avaliacao-teste.json`: resultados no teste.
- `figuras/`: entrada e saída, não linearidade e erro por horizonte.
- `diagnostico/`: estados físicos e medida da não linearidade; fora dos modelos.
