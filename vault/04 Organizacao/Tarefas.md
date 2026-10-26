---
tipo: tarefas
status: atualizado
atualizado: 2026-10-26
tags: [tcc, grupo, tarefas]
---

# Tarefas

## Concluído nesta base

- [x] Analisar o relatório anterior e elaborar o plano para A2.
- [x] Implementar a planta linear, dados independentes, persistência e ARX.
- [x] Executar o primeiro experimento e as oito verificações.
- [x] Guardar resultados, figuras e configuração.
- [x] Criar o vault de acompanhamento do grupo.

## Próxima prioridade

Cronograma e divisão revistos em 17/10: `planejamento/revisao-plano-2026-10-17.md`, na raiz do repositório.

- [ ] Grupo revisar [[E001 - Referencias lineares]] e compreender o protocolo.
- [ ] Discutir o recorte com o orientador.
- [ ] Confirmar sistema operacional e acesso ao PC.
- [x] Validar passagem direta e gradientes do Mamba-3 na implementação escolhida (17/10, PyTorch no Mac).
- [x] Treinar a rede Mamba-3 nos dados do E001 e avaliar nas mesmas origens (17/10, `resultados/mamba3-v1`).
- [ ] Grupo revisar [[E002 - Mamba-3]]: código, protocolo e interpretação.
- [x] Discutir o recorte com o orientador: nova tese em [[Decisao 2026-10 - Tese identificacao vs tradicionais]].
- [x] Executar [[E003 - Duffing]] e [[E004 - Duffing com ruido]] (26/10).
- [x] Redigir o [[Relatorio 2]] em LaTeX (`relatorio-2/`), 26/10.
- [ ] Grupo revisar o relatório 2, a redação da tese e a declaração de uso de assistente de IA; enviar até 20h de 26/10.
- [ ] Ensaiar a [[Apresentacao 2]] (27/10).
- [ ] Grupo revisar E003, E004, `experimento_duffing.py` e `diagnostico_e004.py`.
- [ ] Confirmar se as referências de Ljung e Billings ficam no relatório: citar só o que o grupo consultou.

## Próximos ensaios (cronograma do relatório 2)

- [ ] Fixar as grades de todas as famílias antes de rodar (atrasos até 16, grau até 4), sem conhecimento da planta.
- [ ] Estimar o estado inicial de ARX/NARX com a mesma janela de 256 medições.
- [ ] NARX por erro de simulação e com seleção de termos (FROLS); GRU como referência neural.
- [ ] Treino mais longo da rede e perda de vários passos.
- [ ] Conjunto de teste novo, reservado; intervalos por bootstrap.

## Frentes propostas, responsáveis a combinar

| Frente | Entrega | Responsável |
|---|---|---|
| Dados e referência | Entender simulador, revisar dados e ARX | A definir pelo grupo |
| Modelo neural | Ambiente, implementação, treinamento e diagnóstico | A definir pelo grupo |
| Avaliação e relatório | Métricas, figuras, integração da redação e apresentação | A definir pelo grupo |

Todos devem entender o alinhamento temporal e a diferença entre previsão e simulação. A divisão proposta não atribui responsabilidades aos integrantes sem acordo.

## Extensões depois do principal

- [ ] Comparação com Mamba-2.
- [ ] Uma rodada de ruído ou não linearidade.
- [ ] Estudo de custo computacional sob regime validado.
- [ ] Integração com MPC de referência em novembro, se viável.

Atualizar [[Estado atual]] e [[Diario]] quando uma tarefa mudar de estágio.

