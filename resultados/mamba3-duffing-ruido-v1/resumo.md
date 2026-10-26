# Mamba-3 nos dados do E004

Execução local concluída. Oscilador de Duffing (mola cúbica) simulado, com ruído de medição gaussiano de desvio 0,02 m; mesmos dados, normalização, origens e métricas do E004.

Implementação: Mamba-3 SISO em PyTorch, a partir das funções de referência de `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`, float32, dispositivo `mps`.
Rede: Linear(2→64), 1 bloco(s) Mamba-3 com RMSNorm e residual, Linear(64→1); 34949 parâmetros.

## Escolha na validação

Critério: média, entre as sementes, do menor RMSE da trajetória de 50 passos na validação. Selecionada: **lr3e-3**.

| Configuração | Semente | Melhor época | RMSE validação, 50 passos [m] | Tempo de treino [s] | Situação |
|---|---:|---:|---:|---:|---|
| lr1e-3 | 1 | 270 | 2.488e-02 | 240 | concluido |
| lr1e-3 | 2 | 300 | 2.925e-02 | 242 | concluido |
| lr1e-3 | 3 | 240 | 2.650e-02 | 241 | concluido |
| lr3e-3 | 1 | 270 | 2.466e-02 | 238 | concluido |
| lr3e-3 | 2 | 270 | 2.566e-02 | 234 | concluido |
| lr3e-3 | 3 | 300 | 2.415e-02 | 237 | concluido |
| lr1e-2 | 1 | 290 | 2.526e-02 | 248 | concluido |
| lr1e-2 | 2 | 300 | 2.689e-02 | 262 | concluido |
| lr1e-2 | 3 | 290 | 2.282e-02 | 260 | concluido |

## Teste

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
| Mamba-3 (semente 1) | 1 | 2.218e-02 | 2.218e-02 | 7.853e-02 |
| Mamba-3 (semente 1) | 10 | 2.330e-02 | 2.238e-02 | 7.924e-02 |
| Mamba-3 (semente 1) | 50 | 2.596e-02 | 2.428e-02 | 8.595e-02 |
| Mamba-3 (semente 2) | 1 | 2.260e-02 | 2.260e-02 | 8.001e-02 |
| Mamba-3 (semente 2) | 10 | 2.468e-02 | 2.349e-02 | 8.317e-02 |
| Mamba-3 (semente 2) | 50 | 3.101e-02 | 2.750e-02 | 9.734e-02 |
| Mamba-3 (semente 3) | 1 | 2.220e-02 | 2.220e-02 | 7.860e-02 |
| Mamba-3 (semente 3) | 10 | 2.245e-02 | 2.243e-02 | 7.941e-02 |
| Mamba-3 (semente 3) | 50 | 2.714e-02 | 2.425e-02 | 8.586e-02 |

Mamba-3, média entre sementes:

| Horizonte | Média do RMSE da trajetória [m] | Desvio-padrão [m] | Mínimo [m] | Máximo [m] |
|---:|---:|---:|---:|---:|
| 1 | 2.233e-02 | 2.36e-04 | 2.218e-02 | 2.260e-02 |
| 10 | 2.277e-02 | 6.27e-04 | 2.238e-02 | 2.349e-02 |
| 50 | 2.534e-02 | 1.86e-03 | 2.425e-02 | 2.750e-02 |

Mesmas previsões, comparadas com a posição sem ruído (`metricas-teste-sem-ruido.csv`):

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
| Mamba-3 (semente 1) | 1 | 9.127e-03 | 9.127e-03 |
| Mamba-3 (semente 1) | 10 | 1.197e-02 | 1.054e-02 |
| Mamba-3 (semente 1) | 50 | 1.591e-02 | 1.374e-02 |
| Mamba-3 (semente 2) | 1 | 9.487e-03 | 9.487e-03 |
| Mamba-3 (semente 2) | 10 | 1.478e-02 | 1.270e-02 |
| Mamba-3 (semente 2) | 50 | 2.417e-02 | 1.868e-02 |
| Mamba-3 (semente 3) | 1 | 8.792e-03 | 8.792e-03 |
| Mamba-3 (semente 3) | 10 | 1.161e-02 | 1.065e-02 |
| Mamba-3 (semente 3) | 50 | 1.732e-02 | 1.380e-02 |

## Como ler

As referências clássicas foram recalculadas a partir dos modelos gravados no E004 e conferidas contra as métricas daquela execução. As previsões da rede usam as mesmas origens e os mesmos alvos.
Em cada origem, a rede é aquecida com as 256 medições até a origem, inclusive, e depois realimenta as próprias previsões; saídas futuras reais não entram.
Com ruído de medição, o NARX polinomial é ajustado pelo erro de equação com regressores ruidosos. A tabela principal mede o erro contra a posição medida, com ruído; a tabela sem ruído mede o erro contra a posição verdadeira da planta.
Com três sementes, a dispersão é indicativa; não sustenta uma conclusão estatística ampla.
Tempos de treino desta implementação de referência; não representam os kernels dos autores.

## Artefatos

- `config.json`, `selecao-validacao.json`: configuração, curvas e escolha.
- `modelos/`: pesos do melhor checkpoint de cada execução.
- `metricas-teste.csv`, `avaliacao-teste.json`, `resumo-sementes.json`: resultados no teste.
- `figuras/`: curvas de treino, previsão de um passo, simulação de 50 passos e erro por horizonte.
- `execucao.json`: versões, hashes e dispositivo.
