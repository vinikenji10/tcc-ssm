---
tipo: bibliografia
status: selecao-inicial
atualizado: 2026-10-12
tags: [tcc, bibliografia]
---

# Leituras

## Prioridade para esta fase

| Material | Para que ler | Registro que devemos produzir |
|---|---|---|
| [Mamba Sequence Modeling meets Model Predictive Control — v2](https://arxiv.org/abs/2604.13857v2), Cevaal, de Jong e Lazar | Conectar o preditor à formulação de controle | Entrada/saída, horizonte, inicialização e limites |
| [Mamba-3](https://arxiv.org/abs/2603.15569), Lahoti et al. | Compreender a arquitetura que será avaliada | Recorrência e mecanismos realmente implementados |
| [Transformers are SSMs — Mamba-2](https://arxiv.org/abs/2405.21060), Dao e Gu | Preparar comparação entre versões | Diferenças de estrutura e condições da comparação |
| [From system identification to sequence models](https://bonassifabio.github.io/talks/reglermote2025), materiais de Bonassi | Aproximar a linguagem das duas áreas | Conceitos de identificação associados a SSMs neurais |

O trabalho sobre MPC foi revisado em 28/09/2026. Não assumir que sua formulação ou resultados foram validados para Mamba-3 no nosso sistema.

## Base do relatório anterior

- Gu, Goel e Ré. [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), S4.
- Gu e Dao. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
- Gu et al. [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669).
- Dao et al. [FlashAttention](https://arxiv.org/abs/2205.14135).

As referências completas do primeiro relatório estão em [[Relatorio 1.pdf]].

## Implementação e documentação

- [Repositório oficial state-spaces/mamba](https://github.com/state-spaces/mamba).
- [Bloco Mamba-3](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py).
- Fontes sobre hardware e adaptações em [[Infraestrutura]].

## Como registrar uma leitura

Anotar pergunta, método, condições do experimento, conclusão dos autores e o que ela permite ou não concluir sobre o nosso TCC. Registrar dúvidas e decisões que a leitura pode informar.

Resultados da literatura permanecem resultados da literatura. Nossas métricas precisam vir de uma execução identificada em [[Registro de experimento]].

