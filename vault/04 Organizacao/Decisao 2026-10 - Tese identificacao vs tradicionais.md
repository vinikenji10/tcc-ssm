---
tipo: decisao
status: acordada-com-orientador
data: 2026-10-26
atualizado: 2026-10-26
tags: [tcc, decisao, tese]
---

# Decisão — tese: identificação por SSMs comparada a métodos tradicionais

- **Data do registro:** 26/10/2026. A conversa com o orientador aconteceu antes dessa data; a data exata não foi registrada.
- **Situação:** acordada com o orientador (relato de Vinicius).
- **Participantes que confirmaram:** grupo e professor Pait, segundo Vinicius. Gustavo e Ian devem confirmar a redação da tese.
- **Pergunta ou problema:** o grupo ainda não tinha uma tese a defender; o objetivo "identificação e controle" era amplo demais.

## Escolha e motivo

O TCC deixa de ter o controle preditivo (MPC) como objetivo e passa a focar em **identificação de sistemas via SSMs (Mamba-3), comparada a métodos tradicionais**, verificando se é uma estratégia viável e interessante. O MPC fica como motivação e extensão opcional.

O orientador pediu que o [[Relatorio 2]] tenha três partes:

1. Um breve resumo do artigo principal encontrado: Cevaal, de Jong e Lazar, *Mamba Sequence Modeling meets Model Predictive Control* ([arXiv:2604.13857](https://arxiv.org/abs/2604.13857)).
2. O que o grupo fez de fato (ensaios).
3. A tese do trabalho.

## Evidências e incertezas

- O artigo compara Mamba apenas com outras redes (LSTM e Transformer), sem nenhum método clássico de identificação. Essa é a lacuna que o novo recorte ocupa.
- [[E003 - Duffing]] e [[E004 - Duffing com ruido]] foram executados em 26/10 para levar a comparação a uma planta não linear e com ruído.
- A redação exata da tese está no relatório 2 (`relatorio-2/relatorio-2.tex`) e deve ser validada pelo grupo e pelo orientador.

## Consequências e próximo marco

- Toda comparação nova inclui uma referência tradicional justa: estrutura adequada e ajuste adequado ao ruído, escolhida pelo mesmo critério de validação.
- O cronograma de novembro troca as etapas de MPC por ensaios que testam a tese: referências tradicionais mais fortes, não linearidade não polinomial, ruído, quantidade de dados e um benchmark real (Silverbox).

Veja [[Decisoes e propostas]] e [[Cronograma]].
