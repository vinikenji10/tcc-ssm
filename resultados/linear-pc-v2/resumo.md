# Primeiro experimento de identificação

Execução local concluída. Planta linear simulada; sinais sem ruído.

Mamba-3 e MPC ainda não foram executados.

ARX selecionado somente pela validação: **ARX(2,2)**, com 5 parâmetros.

| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |
|---|---:|---:|---:|---:|
| Persistência | 1 | 4.195e-02 | 4.195e-02 | 8.838e-02 |
| Persistência | 10 | 4.022e-01 | 2.534e-01 | 5.339e-01 |
| Persistência | 50 | 6.582e-01 | 6.839e-01 | 1.441e+00 |
| ARX(2,2) | 1 | 1.130e-15 | 1.130e-15 | 2.381e-15 |
| ARX(2,2) | 10 | 5.201e-14 | 2.768e-14 | 5.831e-14 |
| ARX(2,2) | 50 | 2.263e-13 | 1.420e-13 | 2.993e-13 |

## O que esse resultado estabelece

O ARX foi ajustado apenas aos sinais de treino e avaliado em trajetórias independentes. Na simulação de vários passos, recebe o histórico observado até a origem e, depois dela, usa exclusivamente suas próprias saídas previstas e as entradas conhecidas.

No caso linear ideal e sem ruído, uma referência ARX de ordem adequada pode reproduzir a dinâmica até a precisão numérica. Erros nessa faixa devem ser descritos como próximos da precisão numérica, sem interpretar diferenças ínfimas entre ordens como ganhos práticos.

O experimento verifica a base de simulação e avaliação. Ele não demonstra desempenho de Mamba-3, superioridade de SSMs neurais, robustez a ruído, identificação não linear ou controle em malha fechada.

## Artefatos

- `dados.npz`: sinais e identificadores separados por trajetória.
- `normalizacao-treino.json`: estatísticas calculadas somente no treino.
- `modelo-arx.json` e `selecao-validacao.json`: ajuste e escolha sem usar o teste.
- `metricas-teste.csv` e `avaliacao-teste.json`: resultados e curva por horizonte.
- `figuras/`: quatro figuras, em PNG e SVG.
- `diagnostico/`: estados físicos e discretização; não utilizados como atributos dos modelos.
- `execucao.json`: versões, hashes e escopo verificado.

## Próximo passo

Adicionar o preditor Mamba-3 aos mesmos sinais e origens de avaliação, após validar seu ambiente. Antes de usar redes com memória, implementar e conferir o aquecimento e a regra de reinício de estado. Não comparar uma rede avaliada com medições futuras reais a um ARX simulado sem essas medições.
