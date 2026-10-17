---
tipo: acompanhamento
status: atualizado
atualizado: 2026-10-17
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
| Mamba-3 | Treinado e avaliado no protocolo do E001 em 17/10; três sementes | [[E002 - Mamba-3]] |
| Comparação com Mamba-2 | Planejada | Depende de concluir o modelo principal |
| Ruído e não linearidade | Não avaliados | Extensões posteriores |
| MPC | Não implementado | Depende de validar a previsão |
| Documentação do grupo | Vault criado | [[Como colaborar]] |

## Primeiro resultado

O ARX alcançou RMSE de aproximadamente **1,10 × 10⁻¹³ m** sobre trajetórias previstas de 50 passos na planta linear sem ruído. Descrever como erro próximo da precisão numérica, sem atribuir significado prático às últimas casas decimais.

Isso estabelece uma referência no cenário ideal.

Em 17/10, a rede com um bloco Mamba-3 errou em média **1,16 × 10⁻² m** nas mesmas simulações de 50 passos (três sementes), contra 0,684 m da persistência. Ela aprende a dinâmica, mas não chega à exatidão do ARX na planta linear sem ruído. Veja [[E002 - Mamba-3]].

## Próxima mudança de estágio

A primeira execução do Mamba-3 está registrada. Próximos marcos: revisão do grupo, conversa com o orientador e redação do relatório 2. Extensões possíveis: Mamba-2, ruído ou mola cúbica. Veja [[Tarefas]].

## Como atualizar

Ao concluir uma etapa, inserir aqui o estágio e um link para o registro correspondente. Não marcar como concluído apenas porque existe um plano ou um script ainda não executado.

