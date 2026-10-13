---
tipo: acompanhamento
status: atualizado
atualizado: 2026-10-12
tags: [tcc, acompanhamento]
---

# Estado atual

Atualização baseada nos arquivos e na execução local de 12/10/2026.

| Frente | Estágio | Evidência ou próximo marco |
|---|---|---|
| Pesquisa introdutória | Registrada no relatório 1 | [[Relatorio 1.pdf]] |
| Plano para relatório 2 | Elaborado; recorte a discutir | [[Cronograma]] e [[Decisoes e propostas]] |
| Simulação linear | Implementada e executada | [[E001 - Referencias lineares]] |
| Dados | 30 trajetórias, 61.440 transições | [[Protocolo comum]] |
| Persistência e ARX | Ajustados e avaliados | ARX(2,2) selecionado pela validação |
| Verificações da base | Oito passaram | Testes físicos, causalidade e protocolo |
| Mamba-3 | Ainda não implementado no experimento | [[E002 - Mamba-3]] |
| Comparação com Mamba-2 | Planejada | Depende de concluir o modelo principal |
| Ruído e não linearidade | Não avaliados | Extensões posteriores |
| MPC | Não implementado | Depende de validar a previsão |
| Documentação do grupo | Vault criado | [[Como colaborar]] |

## Primeiro resultado

O ARX alcançou RMSE de aproximadamente **1,10 × 10⁻¹³ m** sobre trajetórias previstas de 50 passos na planta linear sem ruído. Descrever como erro próximo da precisão numérica, sem atribuir significado prático às últimas casas decimais.

Isso estabelece uma referência no cenário ideal. Não há resultado de treinamento ou desempenho do Mamba-3 nesta execução.

## Próxima mudança de estágio

Uma execução do Mamba-3 só será considerada realizada quando houver configuração, código identificado, gradientes verificados, treinamento registrado e avaliação separada no teste. Veja [[Tarefas]].

## Como atualizar

Ao concluir uma etapa, inserir aqui o estágio e um link para o registro correspondente. Não marcar como concluído apenas porque existe um plano ou um script ainda não executado.

