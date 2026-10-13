---
tipo: experimento
id: E002
status: planejado
atualizado: 2026-10-12
tags: [tcc, experimento, mamba]
---

# E002 — Mamba-3

> [!warning] Ainda não executado
> Esta nota descreve o próximo experimento. Não há checkpoint treinado, gráfico ou métrica de Mamba-3 para o TCC.

## Pergunta

Uma rede pequena com Mamba-3 aprende a dinâmica do sistema e sustenta previsões de 50 passos no mesmo teste utilizado pelas referências?

## Configuração inicial proposta

- Sinais numéricos $[u_t,y_t]$, com normalização baseada no treino.
- Projeção para largura 64, um bloco Mamba-3 e saída escalar de regressão.
- Estado interno 64, expansão 2 e variante SISO.
- Janelas inicialmente entre 256 e 512 amostras e lotes pequenos.
- Treinamento do zero; não usar checkpoint de linguagem como se fosse um modelo identificado da planta.

A configuração deve ser ajustada ao suporte da implementação e à memória medida. Veja [[Infraestrutura]].

## Passos de execução

- [ ] Identificar sistema operacional, GPU, versões e commit.
- [ ] Verificar importação e passagem direta com formas corretas.
- [ ] Verificar gradientes finitos e uma atualização de treinamento.
- [ ] Conferir causalidade e equivalência entre sequência e recorrência, quando disponíveis.
- [ ] Implementar cabeça de regressão e normalização.
- [ ] Treinar e escolher checkpoint pela validação.
- [ ] Implementar o aquecimento e reinício do estado.
- [ ] Avaliar nas mesmas trajetórias e origens do [[E001 - Referencias lineares]].
- [ ] Registrar erros, falhas e regime de custo.

## O que guardar

Código e implementação de origem; configuração; versões; sementes; checkpoint; curva de treinamento; escolha na validação; métricas de um passo e simulação; pico de memória; tempo medido sob um regime definido.

Se apenas uma execução for concluída, descrever como resultado preliminar. A comparação com Mamba-2 é uma extensão posterior e precisa usar dados e condições comparáveis.

## Critério para o relatório 2

Entregar uma primeira comparação real, mesmo que a rede não supere ARX na planta linear. Se houver bloqueio técnico, registrar a falha e o trabalho parcial com evidências. Não preencher resultados com números do artigo ou de outra máquina.

Veja [[Tarefas]], [[Cronograma]] e [[Relatorio 2]].

