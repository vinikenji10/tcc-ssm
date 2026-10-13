---
tipo: estudo
status: material-de-apoio
atualizado: 2026-10-12
tags: [tcc, ssm, mamba]
---

# SSMs e Mamba

## A ligação com controle

Um modelo clássico de espaço de estados descreve uma dinâmica interna e sua relação com entradas e saídas:

$$\dot x(t)=Ax(t)+Bu(t),\qquad y(t)=Cx(t)+Du(t).$$

SSMs neurais usam estados internos para processar sequências. A ligação matemática motiva o estudo, mas o estado aprendido não é automaticamente o estado físico da planta.

## S4

O relatório anterior estudou três formas de calcular o componente linear do S4: contínua, recorrente e convolucional. A equivalência com um núcleo fixo exige as hipóteses de linearidade, invariância no tempo e estado inicial correspondente. Consulte [[Relatorio 1.pdf]] e [[Leituras]].

## Mamba

O mecanismo seletivo torna parâmetros dependentes da entrada, alterando como informações são preservadas ao longo da sequência. Isso impede estender diretamente a equivalência com uma convolução fixa ao componente seletivo.

## Mamba-3

O artigo de 2026 propõe mudanças na discretização, atualizações com estrutura complexa e uma formulação interna MIMO. Esses mecanismos motivam o teste do TCC. Os resultados publicados em linguagem e tarefas sintéticas não estabelecem desempenho no nosso problema de identificação.

- O passo interno aprendido da rede não é automaticamente o período físico de amostragem.
- SISO/MIMO da arquitetura interna não corresponde necessariamente ao número de entradas e saídas da planta.
- Usaremos sinais numéricos e uma cabeça de regressão; não precisamos de um modelo de linguagem pré-treinado.

Fonte: [Mamba-3, artigo dos autores](https://arxiv.org/abs/2603.15569).

## Arquitetura e implementação

Uma mesma recorrência pode ser implementada com operadores de referência ou kernels específicos. A fidelidade do cálculo e o suporte ao hardware precisam ser verificados. Uma adaptação para outro dispositivo deve ser identificada como adaptação, e seus tempos não são os tempos dos kernels dos autores.

Veja [[Infraestrutura]] e [[E002 - Mamba-3]].

## Relação com MPC

Uma rede que prevê bem um passo pode acumular erro em simulação. Para MPC, também precisamos definir estado inicial, entradas futuras candidatas, restrições e custo de otimização. A previsão é uma etapa anterior à validação do controlador.

Leitura de ligação: [Mamba Sequence Modeling meets Model Predictive Control, versão 2](https://arxiv.org/abs/2604.13857v2). O artigo não confirma validação de Mamba-3 para este TCC.

