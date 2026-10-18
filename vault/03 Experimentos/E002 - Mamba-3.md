---
tipo: experimento
id: E002
status: executado-localmente
data_execucao: 2026-10-17
atualizado: 2026-10-17
tags: [tcc, experimento, mamba]
---

# E002 — Mamba-3

**Resultado:** sim, com erro de centímetro em 50 passos. No teste, a rede errou em média 1,16 × 10⁻² m ao longo de simulações de 50 passos (três sementes, desvio-padrão 1,0 × 10⁻³ m). A persistência errou 0,684 m e o ARX ficou na precisão numérica (1,1 × 10⁻¹³ m). A rede aprendeu a dinâmica, mas não reproduz a planta linear com a exatidão do ARX. Execução de 17/10/2026; o grupo ainda deve revisar.

## Pergunta

Uma rede pequena com Mamba-3 aprende a dinâmica do sistema e sustenta previsões de 50 passos no mesmo teste utilizado pelas referências?

## Configuração inicial proposta

- Sinais numéricos $[u_t,y_t]$, com normalização baseada no treino.
- Projeção para largura 64, um bloco Mamba-3 e saída escalar de regressão.
- Estado interno 64, expansão 2 e variante SISO.
- Janelas inicialmente entre 256 e 512 amostras e lotes pequenos.
- Treinamento do zero; não usar checkpoint de linguagem como se fosse um modelo identificado da planta.

A configuração deve ser ajustada ao suporte da implementação e à memória medida. Veja [[Infraestrutura]].

## Implementação escolhida em 17/10

O código oficial depende de Triton, que só existe para Linux com GPU NVIDIA. Para não depender do PC, o grupo usa no Mac as funções de referência em PyTorch publicadas pelos autores no arquivo de testes do repositório, no commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`. Elas foram copiadas sem alteração; as projeções do módulo `Mamba3` foram reescritas em volta delas, com os mesmos nomes de parâmetros. Código: `experimentos/mamba3_referencia.py`. Motivo e alternativas: [[Infraestrutura]].

Descrever no relatório como **reimplementação a partir das funções de referência dos autores, em float32**. Só chamar de equivalente ao código oficial depois de comparar as saídas numa GPU NVIDIA, com os mesmos pesos.

### Primeira verificação, 17/10/2026

MacBook Air M5, macOS 26.6.2, Python 3.12.14, torch 2.14.1. Bloco com `d_model=64, d_state=64, expand=2, headdim=64`: duas cabeças e 34.564 parâmetros.

| Medida | CPU | GPU do Mac (MPS) |
|---|---:|---:|
| Erro relativo máximo, recorrência (`step_ref`) contra sequência (`fwd_ref`) | 3,0 × 10⁻⁷ | 2,5 × 10⁻⁶ |
| Erro relativo máximo, aquecer 256 amostras e seguir 256 passos | 1,3 × 10⁻⁶ | 2,7 × 10⁻⁶ |
| Saídas passadas inalteradas ao mudar entradas futuras | sim | sim |
| Gradientes finitos em todos os parâmetros | sim | sim |
| Passo de treino, batch 32 × 512 amostras, mediana | 0,198 s | 0,081 s |
| Avaliação no regime do E001, 545 origens, mediana | 0,65 s | 0,47 s |

Os erros estão no nível da precisão de float32. Numa tarefa de teste (filtro de primeira ordem), 150 atualizações reduziram a perda em mais de dez vezes. Os tempos são desta implementação e não representam os kernels dos autores. Evidência: `resultados/mamba3-verificacao-v1/`.

## Treino e avaliação, 17/10/2026

Script: `experimentos/experimento_mamba3.py`, com `experimentos/config-mamba3.json`. Execução: `resultados/mamba3-v1/`, com código preservado em `fonte/`. Dados, normalização, origens e métricas são os do [[E001 - Referencias lineares]] (mesmo hash de `dados.npz`).

**Rede:** Linear(2→64), um bloco Mamba-3 com RMSNorm e residual, Linear(64→1); 34.949 parâmetros. No instante $t$ entram $u_t$ e $y_t$ normalizados; sai $y_{t+1}$.

**Treino:** erro quadrático de um passo em 260 janelas de 512 amostras (passo de 128) das 20 trajetórias de treino. O estado começa do zero em cada janela e as primeiras 128 posições ficam fora da perda. AdamW, batch 32, decaimento em cosseno, recorte de gradiente 1,0, 300 épocas. GPU do Mac (MPS), float32; cerca de 4 minutos por treino.

**Escolha na validação:** a cada 10 épocas, simulação de 50 passos nas 545 origens da validação; guardado o melhor checkpoint. Três taxas de aprendizado, cada uma com três sementes; vence a menor média do RMSE da trajetória de 50 passos.

| Taxa de aprendizado | Média na validação [m] |
|---|---:|
| 10⁻³ | 2,06 × 10⁻² |
| **3 × 10⁻³ (selecionada)** | **1,50 × 10⁻²** |
| 10⁻² | 2,72 × 10⁻² |

Antes da execução final, um piloto de 100 épocas só na validação levou a incluir a taxa 10⁻² e a manter 300 épocas. Uma execução de verificação do fluxo, com duas épocas, chegou a calcular métricas de teste de uma rede sem treino; nenhuma escolha usou esses números.

**Avaliação:** em cada origem $t$, a rede é aquecida com as 256 medições de $t-255$ a $t$ e depois realimenta as próprias previsões, recebendo as entradas futuras conhecidas, como o ARX. O script confere que persistência e ARX recalculados batem com o E001 e que os alvos da rede são idênticos aos do ARX.

### Resultados de teste

RMSE da trajetória prevista, em metros, nas 545 origens das cinco trajetórias de teste:

| Modelo | 1 passo | 10 passos | 50 passos |
|---|---:|---:|---:|
| Persistência | 4,20 × 10⁻² | 2,53 × 10⁻¹ | 6,84 × 10⁻¹ |
| ARX(2,2) | 8,8 × 10⁻¹⁶ | 2,1 × 10⁻¹⁴ | 1,1 × 10⁻¹³ |
| Mamba-3, semente 1 | 1,02 × 10⁻³ | 5,40 × 10⁻³ | 1,05 × 10⁻² |
| Mamba-3, semente 2 | 1,24 × 10⁻³ | 6,00 × 10⁻³ | 1,25 × 10⁻² |
| Mamba-3, semente 3 | 1,16 × 10⁻³ | 6,20 × 10⁻³ | 1,19 × 10⁻² |
| **Mamba-3, média ± desvio** | **(1,14 ± 0,11) × 10⁻³** | **(5,86 ± 0,42) × 10⁻³** | **(1,16 ± 0,10) × 10⁻²** |

Em 50 passos, o NRMSE da rede fica entre 2,2% e 2,6% do desvio-padrão da posição no teste. A tabela completa, com erro no ponto final, está em `resultados/mamba3-v1/metricas-teste.csv`.

![[E002-simulacao-50-passos.png]]

![[E002-erro-por-horizonte.png]]

![[E002-previsao-um-passo.png]]

![[E002-curvas-treino.png]]

### Interpretação e limites

- A rede aprendeu a relação entre força e posição: em 50 passos erra cerca de 60 vezes menos que a persistência. Ela fica muito longe do ARX, que nessa planta linear sem ruído reproduz a dinâmica exatamente. Isso era esperado e não é uma falha da rede.
- O erro cresce de cerca de 1 mm no primeiro passo para cerca de 1,5 cm no fim de 50 passos. A rede foi treinada só com erro de um passo, e o erro se acumula quando ela realimenta as próprias previsões.
- Na previsão de um passo, os maiores erros aparecem logo depois das trocas de sinal da força, que é um degrau.
- O melhor checkpoint caiu entre as épocas 270 e 300 em todas as sementes, e a validação ainda melhorava devagar. Mais treino, perda de vários passos ou prever a variação de $y$ poderiam reduzir o erro; nada disso foi testado.
- Três sementes dão uma ideia da dispersão, não uma conclusão estatística.
- Os tempos valem para esta implementação de referência na GPU do Mac. Não representam os kernels dos autores nem servem de benchmark.
- Planta linear, simulada e sem ruído; sem MPC.

## Passos de execução

- [x] Identificar sistema operacional, versões e commit. MacBook Air M5; GPU NVIDIA ainda não usada.
- [x] Verificar importação e passagem direta com formas corretas.
- [x] Verificar gradientes finitos e uma atualização de treinamento.
- [x] Conferir causalidade e equivalência entre sequência e recorrência.
- [ ] Comparar com o módulo oficial numa GPU NVIDIA, com os mesmos pesos (opcional até 26/10).
- [x] Implementar cabeça de regressão e normalização.
- [x] Treinar e escolher checkpoint pela validação.
- [x] Implementar o aquecimento e reinício do estado.
- [x] Avaliar nas mesmas trajetórias e origens do [[E001 - Referencias lineares]].
- [x] Registrar erros, falhas e regime de custo. Pico de memória não medido.
- [ ] Grupo revisar código, protocolo e interpretação.

## O que guardar

Código e implementação de origem; configuração; versões; sementes; checkpoint; curva de treinamento; escolha na validação; métricas de um passo e simulação; pico de memória; tempo medido sob um regime definido.

Se apenas uma execução for concluída, descrever como resultado preliminar. A comparação com Mamba-2 é uma extensão posterior e precisa usar dados e condições comparáveis.

## Critério para o relatório 2

Atendido em 17/10 com a execução `mamba3-v1`, pendente de revisão do grupo. Entregar uma primeira comparação real, mesmo que a rede não supere ARX na planta linear. Se houver bloqueio técnico, registrar a falha e o trabalho parcial com evidências. Não preencher resultados com números do artigo ou de outra máquina.

Veja [[Tarefas]], [[Cronograma]] e [[Relatorio 2]].

