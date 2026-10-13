# Registro de avanço experimental - 12/10/2026

**Status:** experimento local executado e verificações concluídas.  
**Uso:** apoio à redação do relatório 2; recorte exploratório proposto no plano.  
**Limite:** Mamba-3 e MPC ainda não foram implementados ou avaliados.

## Atividades realizadas

Foi implementado um simulador do sistema massa-mola-amortecedor linear, com massa de 1 kg, amortecimento de 0,4 N·s/m, rigidez de 4 N/m e período de amostragem de 0,05 s. A entrada é a força aplicada e a saída medida é a posição. A simulação usa discretização exata sob retenção de ordem zero da entrada.

Foram geradas 30 trajetórias independentes, com 2.048 passos cada, totalizando 61.440 transições: 20 trajetórias para treino, cinco para validação e cinco para teste. A força assume valores de -1 ou 1 N, mantidos por intervalos pseudoaleatórios de 5 a 40 amostras. As condições iniciais variam entre trajetórias. Não foi acrescentado ruído de medição nesta execução.

Foram implementadas as referências de persistência e ARX. O ARX foi ajustado exclusivamente aos sinais de entrada e saída de treino. As ordens foram escolhidas pela avaliação de vários passos na validação, com uma regra previamente definida de preferência por modelos menores em caso de empate numérico. A referência selecionada foi ARX(2,2), com quatro coeficientes de atraso e um intercepto.

A avaliação no teste distingue previsão de um passo e simulação de vários passos. Em cada origem, o modelo recebe medições somente até essa origem; dentro do horizonte, realimenta suas próprias saídas previstas. A sequência futura de entradas é conhecida. Foram avaliados horizontes de 1, 10 e 50 passos, correspondentes a 0,05 s, 0,5 s e 2,5 s.

## Resultados obtidos

| Referência | Horizonte | RMSE sobre a trajetória prevista [m] |
|---|---:|---:|
| Persistência | 1 passo | 4,195 × 10⁻² |
| Persistência | 10 passos | 2,534 × 10⁻¹ |
| Persistência | 50 passos | 6,839 × 10⁻¹ |
| ARX(2,2) | 1 passo | 8,832 × 10⁻¹⁶ |
| ARX(2,2) | 10 passos | 2,142 × 10⁻¹⁴ |
| ARX(2,2) | 50 passos | 1,096 × 10⁻¹³ |

O ARX reproduziu a dinâmica com erro próximo da precisão numérica neste cenário linear ideal. Isso é compatível com a ordem do sistema, a ausência de ruído e a geração exata dos dados. O resultado serve como referência para comparar os futuros modelos neurais; não demonstra vantagem de SSMs neurais nem desempenho do Mamba-3.

As quatro figuras produzidas mostram a entrada e a saída, a previsão de um passo, a simulação de 50 passos e a curva de erro por horizonte. As tabelas utilizam todas as origens de teste; os gráficos de trajetórias utilizam trechos predefinidos da primeira trajetória, sem seleção posterior pelo desempenho.

## Verificação e reprodutibilidade

Oito verificações automatizadas passaram, incluindo comparação da discretização com integração RK4 independente, condição de equilíbrio físico, identificação em uma trajetória de teste, rejeição de medições futuras, realimentação de previsões, causalidade das entradas, normalização somente com treino e reprodutibilidade das trajetórias.

Os arquivos guardam a configuração, as sementes e identificadores das trajetórias, os parâmetros estimados, a seleção na validação, as métricas no teste e as versões e hashes da execução. O ambiente de bibliotecas foi criado isoladamente na pasta do projeto. A execução foi em CPU; não foi realizado benchmark de inferência.

## Limitações e trabalho pendente

- O sistema é linear, simulado e sem ruído de medição.
- A comparação atual contém apenas referências clássicas e de persistência.
- Ainda não há teste de rede Mamba-3, comparação entre Mamba-2 e Mamba-3 ou avaliação em planta não linear.
- Os resultados não estabelecem estabilidade em malha fechada ou desempenho de MPC.

O próximo marco é validar o ambiente de Mamba-3, treinar um preditor numérico e inseri-lo no mesmo protocolo de teste. A definição da GPU e o ajuste do recorte com o grupo e o orientador permanecem necessários.

## Arquivos de apoio

- [Resumo e tabela completa](../resultados/linear-v1/resumo.md).
- [Dados de entrada e saída](../resultados/linear-v1/dados.npz).
- [Métricas do teste](../resultados/linear-v1/metricas-teste.csv).
- [Código e instruções](../experimentos/README.md).

