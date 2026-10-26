---
tipo: acompanhamento
status: atualizado
atualizado: 2026-10-26
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
| Planta não linear (Duffing) | Executada em 26/10, com NARX polinomial | [[E003 - Duffing]] |
| Ruído de medição | Um nível (2 cm) avaliado em 26/10 | [[E004 - Duffing com ruido]] |
| Comparação com Mamba-2 | Fora do foco atual | A tese agora compara com métodos tradicionais |
| MPC | Fora do objetivo desde 26/10 | Extensão opcional; veja [[Decisao 2026-10 - Tese identificacao vs tradicionais]] |
| Documentação do grupo | Vault criado | [[Como colaborar]] |

## Primeiro resultado

O ARX alcançou RMSE de aproximadamente **1,10 × 10⁻¹³ m** sobre trajetórias previstas de 50 passos na planta linear sem ruído. Descrever como erro próximo da precisão numérica, sem atribuir significado prático às últimas casas decimais.

Isso estabelece uma referência no cenário ideal.

Em 17/10, a rede com um bloco Mamba-3 errou em média **1,16 × 10⁻² m** nas mesmas simulações de 50 passos (três sementes), contra 0,684 m da persistência. Ela aprende a dinâmica, mas não chega à exatidão do ARX na planta linear sem ruído. Veja [[E002 - Mamba-3]].

Em 26/10, na planta de Duffing, a rede ficou entre o ARX (8,9 cm) e o NARX polinomial, que é quase exato. Com ruído de 2 cm, a execução registrada mostrou a rede muito abaixo do NARX(4,4), mas a revisão encontrou um artefato de grade: com 10 atrasos, o NARX chega a 1,9 cm, equivalente à rede (1,5 cm). A hipótese de interesse (H3) está em aberto. Veja [[E004 - Duffing com ruido]].

## Próxima mudança de estágio

A primeira execução do Mamba-3 está registrada. Próximos marcos: revisão do grupo, conversa com o orientador e redação do relatório 2. Extensões possíveis: Mamba-2, ruído ou mola cúbica. Veja [[Tarefas]].

## Como atualizar

Ao concluir uma etapa, inserir aqui o estágio e um link para o registro correspondente. Não marcar como concluído apenas porque existe um plano ou um script ainda não executado.

