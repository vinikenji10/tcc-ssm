---
tipo: diario
status: atualizado
atualizado: 2026-10-12
tags: [tcc, acompanhamento]
---

# Diário do projeto

## 14/09/2026 — relatório anterior

Entregue o relatório 1 com pesquisa introdutória de S4, Mamba, HiPPO e complexidade. O documento não registrava implementações ou experimentos. Fonte: [[Relatorio 1.pdf]].

## 11/10/2026 — plano para a segunda avaliação

Elaborado o plano de identificação e previsão com massa–mola–amortecedor, ARX e proposta de Mamba-3. Detectada a necessidade de antecipar as primeiras previsões para a entrega de 26/10. Veja [[Cronograma]].

## 12/10/2026 — primeira execução local

Implementados simulador, geração dos dados, persistência, ARX e avaliação. Oito verificações passaram. Executado [[E001 - Referencias lineares]], com ARX de ordem 2 escolhido pela validação. O código e os registros foram preparados com apoio do assistente; o grupo ainda deve revisar seu funcionamento e interpretação.

## 12/10/2026 — infraestrutura identificada e documentação compartilhada

Confirmado localmente o MacBook Air M5 com 16 GB. Informado pelo usuário o PC com RTX 4060 de 8 GB, Ryzen 5 7600X e 32 GB DDR5. O modelo neural ainda não foi executado em nenhuma das máquinas. Criado o vault para acompanhamento do grupo.

## 17/10/2026 - revisão do plano para A2

Constatado que o grupo não avançou desde A1, além da base de 12/10, ainda não revisada. Revistos o cronograma até 27/10, a divisão em três frentes e o caminho do Mamba-3: referência em PyTorch dos autores no Mac, com verificação na RTX 4060 se houver tempo. O HuggingFace não é necessário para o experimento. Veja `planejamento/revisao-plano-2026-10-17.md`.

## 17/10/2026 - bloco Mamba-3 em PyTorch

Montado o bloco Mamba-3 SISO em PyTorch puro a partir das funções de referência dos autores, com apoio do assistente. Verificados na CPU e na GPU do Mac: formatos dos parâmetros iguais aos do módulo oficial, gradientes finitos, causalidade e equivalência entre recorrência e sequência, em nível de precisão de float32. O grupo ainda deve revisar o código. Veja [[E002 - Mamba-3]].

## 17/10/2026 - primeira execução do E002

Treinada a rede com um bloco Mamba-3 nos dados do E001, com apoio do assistente. Três taxas de aprendizado × três sementes, escolha pela simulação de 50 passos na validação, teste nas mesmas 545 origens do ARX. Resultado em 50 passos: (1,16 ± 0,10) × 10⁻² m, contra 0,684 m da persistência e 1,1 × 10⁻¹³ m do ARX. Veja [[E002 - Mamba-3]].

Para próximas entradas, registrar data, o que mudou, resultado verificável e próximo passo. Conversas com orientador devem ter uma nota própria baseada em [[Reuniao]].

