---
tipo: indice
status: atualizado
atualizado: 2026-10-17
tags: [tcc, inicio]
---

# TCC — comece aqui

Investigamos modelos de espaço de estados estruturados para **identificar sistemas dinâmicos a partir de dados** e, depois, avaliar seu uso em controle preditivo.

> [!info] Estado atual
> Temos uma base experimental executada: sistema massa–mola–amortecedor, referências persistência e ARX, dados independentes e gráficos. Em 17/10, uma rede com Mamba-3 foi treinada e avaliada no mesmo protocolo ([[E002 - Mamba-3]]). **MPC ainda não foi executado.**

## Rota de leitura para o grupo

1. [[Visao geral]] — qual é a pergunta do TCC e o recorte proposto.
2. [[Estado atual]] — o que existe e o que ainda falta.
3. [[E001 - Referencias lineares]] e [[E002 - Mamba-3]] — experimentos, resultados e interpretação.
4. [[Previsao e simulacao]] — a distinção que orienta nossa avaliação.
5. [[Tarefas]] e [[Cronograma]] — o próximo trabalho e os prazos.

## Navegação

| Projeto | Estudo | Experimentos | Grupo e entregas |
|---|---|---|---|
| [[Visao geral]] | [[Identificacao de sistemas]] | [[Protocolo comum]] | [[Como colaborar]] |
| [[Estado atual]] | [[SSMs e Mamba]] | [[E001 - Referencias lineares]] | [[Decisoes e propostas]] |
| [[Cronograma]] | [[Previsao e simulacao]] | [[E002 - Mamba-3]] | [[Diario]] |
|  | [[Leituras]] | [[Infraestrutura]] | [[Relatorio 2]] e [[Apresentacao 2]] |

## Resultados em uma figura

![[E002-simulacao-50-passos.png]]

A curva ARX coincide com a saída real nesta planta linear sem ruído, como esperado num cenário ideal de ordem adequada. As três redes Mamba-3 (uma por semente) acompanham a dinâmica e se afastam alguns centímetros perto do fim dos 50 passos. Em média, erram 1,2 cm nessas simulações; a persistência erra 68 cm.

## Como manter este vault útil

- Resultados medidos entram em uma nota de experimento, com configuração e arquivos de evidência.
- Escolhas ainda não discutidas aparecem como propostas; acordos do grupo e do orientador entram em [[Decisoes e propostas]].
- Cada nova execução recebe um identificador e uma pasta de resultados própria.
- Atualizem [[Estado atual]] e [[Tarefas]] quando uma entrega mudar de estágio.

Modelos de notas: [[Registro de experimento]], [[Reuniao]] e [[Decisao]].

