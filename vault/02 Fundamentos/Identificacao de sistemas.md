---
tipo: estudo
status: material-de-apoio
atualizado: 2026-10-12
tags: [tcc, identificacao]
---

# Identificação de sistemas

Identificar um sistema é construir um modelo de sua dinâmica a partir de dados observados. Aqui, temos uma força aplicada $u_t$ e uma posição medida $y_t$; queremos prever posições futuras.

## Planta, dados e modelo são objetos diferentes

- **Planta:** sistema que produz as observações; no E001, é um simulador físico conhecido.
- **Dados:** trajetórias de força e posição produzidas pela planta.
- **Modelo identificado:** preditor ajustado aos dados, sem receber os parâmetros físicos verdadeiros.

Conhecer a planta para gerar o conjunto não significa fornecer sua dinâmica ao algoritmo de ajuste. Esse isolamento está implementado no [[E001 - Referencias lineares]].

## Referência ARX

Um ARX prevê a saída seguinte a partir de atrasos de saídas e entradas:

$$
\hat y_{t+1}=\sum_{i=0}^{n_y-1} a_i y_{t-i}+\sum_{j=0}^{n_u-1} b_j u_{t-j}+b_0^{\mathrm{intercepto}}.
$$

Os coeficientes são estimados por regressão linear. No nosso ARX(2,2), há dois atrasos de saída, dois de entrada e um intercepto, totalizando cinco parâmetros.

O uso de $u_t$ é correto na convenção do experimento: $y_t$ é observado antes de aplicar $u_t$, e o alvo é $y_{t+1}$.

## Por que uma referência clássica?

Ela oferece uma comparação simples e ajuda a verificar se o protocolo está correto. Na planta linear ideal, uma referência de ordem adequada pode ter desempenho excelente. O objetivo científico é avaliar em quais condições modelos diferentes são úteis.

## Três conjuntos de dados

- **Treino:** ajustar os parâmetros.
- **Validação:** escolher configuração, ordem e checkpoint.
- **Teste:** avaliar o modelo após congelar as escolhas.

Trajetórias independentes devem ser separadas antes de extrair janelas. Janelas sobrepostas de uma mesma trajetória não podem aparecer em conjuntos diferentes no nosso protocolo.

Para entender o que significa prever, continue em [[Previsao e simulacao]]. Para os valores concretos, veja [[Protocolo comum]].

