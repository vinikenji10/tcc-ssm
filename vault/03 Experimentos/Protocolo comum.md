---
tipo: protocolo
status: executado-nas-referencias
atualizado: 2026-10-12
tags: [tcc, experimento, protocolo]
---

# Protocolo comum

Este protocolo foi usado para as referências do E001. A adaptação do Mamba-3 ainda precisa implementar e verificar as mesmas condições de avaliação.

## Planta e sinais

$$\dot p=v,\qquad m\dot v=-cv-kp+u,\qquad y=p.$$

| Parâmetro | Valor |
|---|---:|
| Massa | 1 kg |
| Amortecimento | 0,4 N·s/m |
| Rigidez | 4 N/m |
| Amostragem | 0,05 s |
| Entrada | Força entre -1 e 1 N |
| Saída do preditor | Posição em metros |

A planta linear é discretizada por ZOH exata. Estados físicos são guardados apenas para diagnóstico e não entram nos modelos identificados.

## Dados do E001

- Semente principal: `20261012`.
- 20 trajetórias de treino, 5 de validação e 5 de teste.
- 2.048 transições por trajetória, com 2.049 medições de saída.
- Força binária pseudoaleatória, com retenção entre 5 e 40 amostras.
- Posição e velocidade iniciais sorteadas em intervalos de -0,2 a 0,2 nas respectivas unidades.
- Sem ruído de medição nesta execução.
- Estatísticas de normalização calculadas somente no treino; ARX ajustado em unidades físicas.

No instante $t$, entram $u_t$ e $y_t$; o alvo é $y_{t+1}$. Uma observação adicional de saída guarda o efeito da última entrada.

## Escolha do ARX

Foram testadas ordens de saída e entrada em `{2, 4, 8}`, formando nove candidatos. Critério: RMSE da trajetória de 50 passos na validação. Empate dentro de `1e-12` m favorece o menor modelo. O teste não escolhe a ordem.

## Origens de previsão

As origens começam em `t=256`, com espaçamento de 16 amostras e espaço para completar 50 passos. Isso resulta em 109 origens por trajetória de teste e 545 previsões de trajetórias por modelo.

Todos os modelos usam as mesmas origens. ARX lê seus últimos atrasos no histórico observado. Para o modelo neural, a regra de aquecimento e reinício de estado precisa ser explicitamente implementada.

## Métricas

$$\mathrm{RMSE}=\sqrt{\frac1n\sum_i(y_i-\hat y_i)^2},\qquad\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\sigma_y}.$$

O denominador é o desvio-padrão dos alvos avaliados no conjunto correspondente, comum aos horizontes. É uma convenção da métrica, não uma estatística para ajustar o modelo usando o teste. Para uma saída constante, NRMSE é indefinido; RMSE continua disponível.

## Reprodutibilidade

Guardar configuração, sementes, identidades das trajetórias, versões, código, parâmetros e métricas. Novas execuções recebem uma pasta nova. Não descartar falhas ou previsões não finitas silenciosamente.

Veja [[E001 - Referencias lineares]], [[E002 - Mamba-3]] e [[Registro de experimento]].

