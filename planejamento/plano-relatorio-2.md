# Plano de trabalho para o relatório 2 do TCC

**Elaborado em:** 11/10/2026.  
**Status:** proposta de planejamento, a ajustar ao trabalho feito desde setembro e à infraestrutura disponível.  
**Base:** relatório de 14/09/2026 e orientações da disciplina fornecidas na conversa. Este documento não registra execução de experimentos nem resultados novos.

**Atualização em 12/10:** a base experimental com planta linear, persistência e ARX foi implementada e executada localmente. Os resultados estão no [registro desta fase](registro-experimento-2026-10-12.md). Mamba-3 e MPC continuam pendentes. As seções abaixo preservam o planejamento original de 11/10.

## 1. O avanço que a segunda entrega deve mostrar

O relatório anterior apresenta o tema, as três representações do S4, o HiPPO e uma análise preliminar de complexidade. Ele declara que ainda não havia implementação, treinamento, comparação ou simulação de controle. A escolha do sistema, das ferramentas e da referência de comparação permanecia aberta.

A próxima entrega deve tornar a pergunta de pesquisa operacional:

> Um modelo pequeno com blocos Mamba-3 consegue identificar a dinâmica de um sistema simulado a partir de entradas e saídas e prever sua evolução por vários passos? Como seus erros e custos se comparam aos de uma referência clássica, no mesmo conjunto de dados?

Se houver tempo para incluir Mamba-2, acrescentar uma pergunta secundária: o Mamba-3 oferece alguma vantagem neste problema e orçamento de treinamento? Sem essa comparação, evitar atribuir um resultado especificamente à evolução da arquitetura.

A proposta é **priorizar identificação e previsão até 26/10**, mantendo a integração com MPC como etapa de novembro. O cronograma anterior colocava as primeiras previsões entre 19/10 e 01/11: esse marco precisa ser antecipado para aparecer na segunda avaliação.

### Entrega mínima pretendida

- Um sistema escolhido e suas equações, entrada, saída e período de amostragem documentados.
- Dados simulados separados em treino, validação e teste.
- Uma referência ARX ajustada e um modelo pequeno com Mamba-3 treinado, se o ambiente for viável.
- Previsões de um passo e simulações de vários passos no teste.
- Uma tabela de erros e gráficos com análise das limitações.
- Um cronograma atualizado até a avaliação final.

Esse mínimo é uma meta de trabalho proposta, não uma exigência adicional da disciplina. Se Mamba-3 não executar, o relatório deve mostrar os resultados realmente obtidos com as referências e o diagnóstico técnico, identificando claramente a parte pendente.

## 2. Onde o Mamba-3 entra

O artigo foi submetido ao arXiv em **16/03/2026**. Suas mudanças centrais são uma discretização exponencial-trapezoidal, atualizações com estrutura complexa e uma formulação interna MIMO. Os testes publicados concentram-se em tarefas sintéticas e modelagem de linguagem; eles motivam o experimento, mas não estabelecem adequação ao sistema ou ao MPC do TCC. [Artigo Mamba-3](https://arxiv.org/abs/2603.15569).

Usar a **arquitetura como componente de uma rede de regressão para sinais numéricos**, treinada com os dados do sistema. Os checkpoints oficiais são modelos de linguagem; não são necessários para este experimento. [Exemplo de checkpoint oficial](https://huggingface.co/state-spaces/mamba3-siso-187m).

A discretização interna aprendida da rede e o período físico de amostragem são objetos diferentes. Não identificar automaticamente o parâmetro interno de passo da rede com o intervalo de amostragem da planta. Da mesma forma, o estado latente da rede não é, por definição, o estado físico do sistema.

A formulação MIMO interna da arquitetura pode ser utilizada mesmo com uma planta física SISO. Aqui, começar com **Mamba-3 SISO** para reduzir as dependências e o número de escolhas; deixar MIMO para uma extensão.

### Viabilidade da implementação

O repositório oficial disponibiliza `Mamba3` e orienta instalar a versão atual a partir do código-fonte. Seu caminho documentado usa Linux e GPU NVIDIA/CUDA. Não assumir que uma instalação de PyTorch em um Mac executará esses kernels. [Repositório oficial](https://github.com/state-spaces/mamba).

Há duas verificações distintas: executar sequências com gradientes e executar uma amostra por vez com estado persistente. O módulo usa Triton para SISO, TileLang para MIMO e CuTe DSL para `step`. A documentação de `step` informa teste apenas em H100; outras GPUs precisam de verificação específica. Portanto, sucesso no treinamento não confirma disponibilidade de inferência incremental. [Código oficial do bloco](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py).

**Decisão até 14/10:** registrar GPU, memória, sistema operacional, versões e commit da implementação; executar passagem direta e cálculo de gradientes. Limitar a investigação inicial do ambiente a um dia de trabalho efetivo. Se continuar inviável, avançar com dados, ARX e uma GRU pequena em PyTorch, deixando Mamba-3 como avaliação pendente. Não apresentar GRU ou uma adaptação simplificada como execução do Mamba-3 oficial.

Se o processamento por sequências funcionar, mas `step` não, ainda é possível estudar previsão por recomputação de prefixos causais. Esse regime deve ser nomeado e medido como tal; seus tempos não representam inferência recorrente com estado persistente.

## 3. Primeiro sistema proposto

Usar um **massa-mola-amortecedor linear**, com posição medida e força como entrada:

\[
\dot p=v,\qquad m\dot v=-cp-kp+u,\qquad y=p.
\]

Configuração inicial proposta, ajustável antes de congelar os dados:

| Quantidade | Valor | Finalidade |
|---|---:|---|
| Massa \(m\) | 1 kg | Simplificar as equações |
| Amortecimento \(c\) | 0,4 N·s/m | Manter dinâmica oscilatória amortecida |
| Rigidez \(k\) | 4 N/m | Criar um sistema simples de segunda ordem |
| Amostragem \(T_s\) | 0,05 s | Registrar a evolução temporal |
| Entrada \(u\) | entre -1 e 1 N | Delimitar a região do estudo |
| Saída observada | posição \(p\) | Identificar por entrada e saída |

Simular a planta por discretização ZOH exata do modelo linear. Os estados verdadeiros podem ser salvos para verificar o simulador, mas não devem ser fornecidos aos modelos nesta comparação. As equações conhecidas geram os dados; ARX e redes são ajustados a partir desses dados, sem receber os parâmetros físicos.

O sistema linear serve para verificar a implementação e o protocolo. Uma referência linear bem ajustada pode obter erro muito baixo: não há motivo para exigir que Mamba-3 a supere. Essa observação é um resultado útil.

**Extensão posterior:** adicionar uma força de mola cúbica, \(m\dot v=-cv-kp-\alpha p^3+u\), começando com \(\alpha=0,5\,\mathrm{N/m^3}\). Verificar se essa intensidade produz efeito mensurável na região excitada; não assumir dificuldade não linear apenas pela presença do termo. Validar a integração numérica antes de treinar. Essa extensão só começa após fechar o experimento linear.

## 4. Dados e definição da tarefa

### Conjunto inicial proposto

Gerar 30 trajetórias independentes com 2.048 passos cada: 20 para treino, 5 para validação e 5 para teste. Usar sementes e entradas diferentes em cada trajetória. Esse tamanho é um ponto inicial de engenharia, não uma afirmação de suficiência estatística.

Usar uma entrada binária pseudoaleatória entre -1 e 1 N, mantida por intervalos sorteados entre 5 e 40 amostras. Verificar visualmente se ela excita a dinâmica; documentar o gerador e a distribuição dos intervalos. Variar também as condições iniciais, por exemplo \(p_0\in[-0,2;0,2]\) m e \(v_0\in[-0,2;0,2]\) m/s. Reservar degraus e uma entrada multisseno para testes adicionais, separados do teste principal.

Começar **sem ruído**, para detectar erros de alinhamento ou implementação. Acrescentar ruído de medição apenas em uma rodada separada, com intensidade declarada e os modelos reavaliados no mesmo protocolo.

Regras do conjunto:

- Dividir por trajetória antes de extrair janelas; não distribuir janelas sobrepostas da mesma trajetória entre treino e teste.
- Calcular normalização apenas no treino e reaplicá-la aos demais conjuntos.
- Não fornecer velocidade verdadeira, estados futuros ou parâmetros da planta às redes.
- Usar validação para escolher configurações e checkpoint; congelar as escolhas antes da avaliação final no teste.
- Guardar configuração, sementes e identificadores das trajetórias para reproduzir os resultados.

### Previsão de um passo

No instante \(t\), o modelo recebe a entrada aplicada \(u_t\), a saída já medida \(y_t\) e o histórico disponível, e prevê \(y_{t+1}\):

\[
s_t=f_\theta(s_{t-1},[u_t,y_t]),\qquad \hat y_{t+1}=g_\theta(s_t).
\]

O índice da saída alvo deve ser explicitamente deslocado em um passo. Medir erro de um passo usando as medições passadas reais é válido, desde que esse regime seja declarado.

### Simulação de vários passos

Aquecer o modelo com 256 amostras reais anteriores à origem da previsão. A partir da origem, fornecer a sequência futura de entradas conhecida no experimento e realimentar as saídas previstas. Não inserir saídas reais futuras durante o horizonte:

\[
\hat y_{t+1}\rightarrow f_\theta(\ldots,[u_{t+1},\hat y_{t+1}])\rightarrow\hat y_{t+2}\rightarrow\cdots
\]

Avaliar horizontes \(H=1,10,50\), equivalentes a 0,05 s, 0,5 s e 2,5 s nesta amostragem. Usar as mesmas origens e o mesmo histórico observado para todos os modelos. A referência ARX utiliza os atrasos necessários dentro desse histórico.

Se for necessário recomputar prefixos para o Mamba-3, manter o histórico de medições até a origem e acrescentar apenas os valores previstos depois dela. Não substituir inadvertidamente esses valores por medições reais. Reiniciar o estado entre trajetórias independentes.

## 5. Modelos e experimentos, em ordem

| Prioridade | Experimento | Comparação e resultado a produzir |
|---|---|---|
| 0 | Verificar dados e causalidade | Simulador consistente; formas dos tensores; alinhamento; gradientes finitos; mudança de entradas futuras não altera previsões anteriores |
| 1 | Identificação linear | Persistência, ARX e Mamba-3 SISO; erros de um passo e de simulação no teste |
| 2 | Comparação entre versões | Adicionar Mamba-2 no mesmo sistema, dados e tarefa; registrar parâmetros e orçamento de treinamento |
| 3 | Robustez | Rodada separada com ruído ou entradas de teste diferentes; explicitar qual fator mudou |
| 4 | Extensão não linear | Repetir protocolo na planta com mola cúbica, após validar simulador e intensidade da não linearidade |
| 5 | Custo e capacidade | Variar comprimento de sequência e dimensão do estado; MIMO apenas se houver tempo e suporte |

**Até outubro, priorizar 0 e 1; incluir 2 se viável.** Escolher uma extensão por vez após essa entrega.

### Referências

- **Persistência:** \(\hat y_{t+1}=y_t\); na simulação, manter a saída inicial constante. Ajuda a detectar se um pequeno erro de um passo é apenas consequência da amostragem fina.
- **ARX:** regressão linear sobre atrasos de entradas e saídas. Usar ordens de saída e entrada em uma grade pequena, por exemplo 2, 4 e 8, selecionadas por validação. Incluir \(u_t\) ao prever \(y_{t+1}\). Verificar condicionamento da regressão e comportamento da simulação, não apenas o erro de ajuste.
- **Mamba-2:** comparação preferida para investigar a evolução da família; não obrigatório para a primeira demonstração.
- **GRU:** alternativa se os kernels do Mamba-3 estiverem bloqueados ou comparação adicional em novembro. Não ampliar agora para S4, LSTM e Transformer simultaneamente.

### Rede numérica inicial

Projeção linear dos dois sinais \([u_t,y_t]\) para uma representação interna, um bloco Mamba-3 com normalização e conexão residual, e projeção final para uma saída escalar. Treinar do zero com erro quadrático de um passo.

Configuração inicial proposta: `d_model=64`, `d_state=64`, `expand=2`, `headdim=64`, `is_mimo=False`, `chunk_size=64`. Validar essas formas no commit e hardware escolhidos. Começar com janelas de 512 amostras, reiniciando o estado em cada janela e excluindo as primeiras 128 posições da perda para acomodar o aquecimento. Registrar essa escolha; ela não equivale a treinamento com estado mantido durante uma trajetória inteira.

Usar uma configuração pequena de otimizador, por exemplo AdamW com taxa inicial \(10^{-3}\), e acompanhar a perda de validação. Fixar um teto de épocas e de tempo após medir uma execução curta. Escolher o checkpoint pela métrica de validação definida antes dos testes; preferencialmente erro de simulação em \(H=50\). Se houver instabilidade, verificar alinhamento, normalização, precisão e gradientes antes de aumentar o modelo.

Rodar três sementes de treinamento quando o tempo permitir. Informar cada resultado e sua dispersão; três sementes não sustentam por si só uma conclusão estatística ampla. Se houver apenas uma execução, identificar os resultados como preliminares.

Na comparação com Mamba-2, começar com a mesma largura, profundidade e dimensão do estado, usando a mesma estrutura externa. Contar os parâmetros reais: dimensões iguais não garantem número igual de parâmetros. Ajustar orçamento em uma rodada adicional se a diferença for relevante. Não atribuir diferenças à arquitetura quando dados, ajuste ou capacidade estiverem confundidos.

## 6. O que medir e apresentar

### Qualidade

Calcular RMSE em unidades físicas e NRMSE normalizado pelo desvio-padrão de todas as saídas de teste avaliadas no cenário:

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2},\qquad
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\sigma_{y,\mathrm{teste}}}.
\]

O denominador do teste é usado somente para expressar a métrica após congelar as escolhas; não serve para normalizar entradas da rede nem selecionar modelos. Declarar a convenção, pois NRMSE tem definições diferentes. No caso de saída constante, reportar RMSE e identificar o denominador nulo.

Para cada horizonte, separar o erro no ponto final do horizonte e o erro médio sobre a trajetória prevista. Produzir também a curva de RMSE em função do passo futuro, de 1 a 50. Registrar falhas numéricas e trajetórias com crescimento anormal sem descartá-las silenciosamente.

### Custo

Registrar número de parâmetros, tempo de treinamento e configuração computacional. Se viável, medir tempo para processar sequências e pico de memória da GPU. Medir atualização por amostra com estado persistente apenas quando esse caminho funcionar e for validado.

Ao medir GPU, aquecer os kernels e excluir compilação inicial; sincronizar antes e depois da região cronometrada ou usar eventos CUDA. Repetir medições, registrar mediana e percentil 95, batch, precisão e quais operações entraram no tempo. Separar memória de pesos, estados persistentes e pico de ativações; a memória reservada pelo alocador não é a mesma coisa que memória utilizada.

Para estudar escalabilidade, variar \(L=256,512,1024,2048\) mantendo as demais dimensões fixas. Essa rodada mede custo e não demonstra, por si só, qualidade com históricos mais longos. Não comparar ARX em CPU e Mamba em GPU como se a diferença fosse somente arquitetural.

### Figuras e tabela para a entrega

1. Esquema: força e posição medidas → modelo identificado → posição futura prevista.
2. Uma trajetória de entrada e saída, com amostragem e unidades.
3. Saída real e previsão de um passo, em trecho de teste.
4. Saída real e simulação de 50 passos, com origem e histórico de aquecimento indicados.
5. Erro em função do horizonte, reunindo modelos e sementes disponíveis.
6. Tabela com RMSE/NRMSE, parâmetros, tempo de treinamento e regime de execução.

Pode reunir figuras para manter o relatório conciso. Informar resultados ausentes como “não medido”, sem preencher com estimativas ou números do artigo.

**Interpretação:** erro baixo com medições reais a cada passo não garante boa simulação; boa simulação em alguns cenários não garante estabilidade em malha fechada; custo assintótico favorável não garante menor latência no hardware disponível. Se ARX tiver melhor desempenho, explicar em quais condições isso ocorreu.

## 7. Cronograma até a segunda avaliação

Pelas orientações fornecidas, a entrega é **26/10/2026, segunda-feira, até 20h**, e a apresentação remota é **27/10, terça-feira, a partir das 8h**. O horário individual e o link dependem do Moodle. Conferir eventuais atualizações da disciplina.

| Data | Trabalho | Evidência esperada ao terminar |
|---|---|---|
| 11 a 13/10 | Levantar o que já foi feito; discutir recorte com orientador; definir sistema, tarefa e funções do grupo | Protocolo de uma página e responsabilidades propostas |
| Até 14/10 | Verificar ambiente Mamba-3; escolher caminho principal ou alternativo | Log de execução com gradientes, ou diagnóstico do bloqueio e alternativa definida |
| 14 a 16/10 | Gerar dados, validar simulador, ajustar ARX e persistência | Dados congelados, primeiro gráfico e erros das referências |
| 17 a 20/10 | Treinar rede pequena; corrigir implementação; adicionar Mamba-2 se o principal já funcionar | Configuração e checkpoint; perdas de treino/validação |
| 21 e 22/10 | Congelar escolhas; executar teste; repetir sementes conforme orçamento | Tabela de resultados, curvas de horizonte e logs |
| 23 e 24/10 | Redigir método, resultados e discussão; preparar apresentação | Primeira versão completa do relatório e roteiro de 10 minutos |
| 25/10 | Revisar com o grupo; buscar retorno do orientador com antecedência | Versão revisada, referências e cronograma final |
| 26/10 | Conferir PDF e arquivos; enviar ao Moodle e encaminhar ao orientador até 20h | Comprovante de entrega e versão final enviada |
| 27/10 | Apresentar por 10 minutos e responder por 5; solicitar envio da nota do orientador à banca | Feedback registrado para orientar novembro |

Se o modelo não produzir previsões até 20/10, priorizar corrigir ou diagnosticar esse experimento e fechar o relatório. Não acrescentar MIMO, nova planta ou novo controlador nessa semana. Atividades previstas devem aparecer como previstas; execução parcial ou falha também deve ser descrita com evidência.

## 8. Estrutura proposta para o relatório 2

Manter o título geral e explicitar no resumo que esta fase avalia identificação e previsão. Sugestão de 6 a 8 páginas mais referências, ajustável ao conteúdo; não há limite de páginas nas orientações fornecidas.

1. **Resumo dos resultados anteriores:** cerca de meia página sobre pesquisa introdutória, três visões do S4, HiPPO e complexidade. Registrar o estágio anterior sem repetir todas as derivações.
2. **Objetivo e delimitação desta fase:** pergunta experimental, recorte escolhido e relação com o MPC posterior.
3. **Fundamentação necessária:** ligação entre sistemas físicos e SSMs neurais, diferenças relevantes do Mamba-3 e literatura de Mamba para controle. Separar resultados publicados de hipóteses do grupo.
4. **Metodologia:** planta, discretização, dados, separação, normalização, entrada/saída da rede, referências, treinamento, aquecimento, previsão de um passo e simulação de vários passos.
5. **Atividades realizadas e resultados obtidos:** somente o que foi executado, com gráficos, métricas, configurações e dificuldades. Explicar escolhas feitas desde a primeira entrega.
6. **Discussão e limitações:** interpretação dos erros, condições da comparação, infraestrutura e requisitos ainda pendentes para MPC.
7. **Cronograma detalhado até dezembro:** entregas verificáveis e dependências, incorporando o estágio alcançado.
8. **Referências.**

O principal avanço narrativo deve ser: **conceitos iniciais → problema definido → implementação → primeira evidência experimental → próxima decisão**.

### Apresentação de 10 minutos

| Slide | Conteúdo | Tempo |
|---|---|---:|
| 1 | Problema e estágio anterior | 1 min |
| 2 | Pergunta da fase e motivo para avaliar Mamba-3 | 1 min |
| 3 | Sistema, dados e referências | 2 min |
| 4 | Diferença entre previsão de um passo e simulação | 1 min |
| 5 | Resultados e limitações | 3 min |
| 6 | Próximas entregas e cronograma | 2 min |

Reservar a explicação matemática mais extensa para perguntas ou material de apoio. Os gráficos devem indicar modelos, unidades e regime de previsão.

## 9. Caminho até a avaliação final

| Período | Resultado a perseguir |
|---|---|
| 28/10 a 08/11 | Incorporar feedback; consolidar repetibilidade; completar comparação com Mamba-2 e uma extensão escolhida |
| 09 a 15/11 | Avaliar não linearidade e robustez; verificar se a qualidade de simulação é suficiente para iniciar MPC |
| 16 a 22/11 | Implementar MPC de referência com o modelo linear e testar, em simulação, a integração do modelo aprendido se viável |
| 23 a 29/11 | Comparar rastreamento, esforço de controle, violações de restrições e tempo total de solução; encerrar novos experimentos |
| 30/11 a 06/12 | Consolidar relatório final; revisar; ensaiar apresentação presencial de 15 minutos |
| 07/12 | Prazo de relatório inferido dos mesmos procedimentos: até 20h, sujeito à confirmação no Moodle |
| 08/12 | Apresentação presencial a partir das 9h; 15 minutos e 5 minutos de perguntas |

### Condições para começar o MPC com a rede

- O preditor utiliza apenas medições disponíveis e entradas candidatas futuras.
- A avaliação de vários passos foi feita sem saídas futuras reais.
- O estado ou histórico inicial do preditor tem uma regra definida.
- A otimização consegue avaliar trajetórias candidatas com um custo aceitável; se houver gradientes, o caminho de diferenciação foi verificado.
- Cada trajetória candidata parte da mesma condição inicial: copiar o estado antes das avaliações e evitar que uma candidata altere a seguinte.
- Há um controlador de referência para comparar comportamento e custo.

Usar horizonte inicial de 10 a 20 passos, restrições de entrada e um custo quadrático de rastreamento e esforço, explicitando pesos. Medir o tempo do controlador completo, incluindo previsão e otimização. Só discutir execução em tempo real se as medições sustentarem a comparação com o período de controle escolhido. Simulações bem-sucedidas não constituem prova formal de estabilidade.

Caso a integração não seja viável, registrar o estágio e as razões, conforme a delimitação já admitida no relatório anterior. Levar essa decisão ao orientador durante novembro, para evitar uma mudança tardia de escopo.

A avaliação final tem peso maior: \(M=(A_1+A_2+2A_3)/4\). Usar a segunda entrega para consolidar uma base que permita resultados mais fortes em dezembro. Em cada avaliação, a nota combina banca e orientador, conforme as orientações recebidas.

## 10. Divisão de trabalho e estudo

Distribuição proposta por função, ainda sem atribuir nomes:

| Frente | Responsabilidade | Entrega que os outros usam |
|---|---|---|
| A | Planta, geração de dados e ARX | Dados congelados, simulador e referência de erro |
| B | Ambiente, rede Mamba-3 e treinamento | Configuração, checkpoint e diagnóstico técnico |
| C | Protocolo de avaliação, gráficos e redação | Métricas reproduzíveis, figuras e relatório integrado |

Todos devem entender o alinhamento temporal, a separação dos dados e a diferença entre previsão e simulação. Integrar os resultados diariamente durante a semana de experimentos, usando os mesmos arquivos de dados e convenções de índices. A distribuição é uma proposta para discussão entre os três integrantes.

### Leituras por utilidade

1. **Mamba Sequence Modeling meets Model Predictive Control**, Cevaal, de Jong e Lazar: publicado inicialmente em abril, com versão 2 em 28/09/2026. Ler objetivo, formulação do preditor, entrada/saída, metodologia e limitações. O trabalho trata de Mamba para MPC; não assumir que sua formulação foi validada para Mamba-3. Serve como ponte metodológica e antecedente, sem transferir seus resultados para este TCC. [Artigo, versão 2](https://arxiv.org/abs/2604.13857v2).
2. **Mamba-3: Improved Sequence Modeling using State Space Principles**, Lahoti et al.: priorizar a seção metodológica e identificar o que será efetivamente implementado. [Artigo](https://arxiv.org/abs/2603.15569).
3. **Transformers are SSMs**, Dao e Gu: consultar para compreender Mamba-2 e justificar a comparação entre versões. [Artigo Mamba-2](https://arxiv.org/abs/2405.21060).
4. **From system identification to sequence models**, Bonassi: apoio para conectar a terminologia de identificação à de SSMs neurais. [Página e materiais do autor](https://bonassifabio.github.io/talks/reglermote2025).

Uma contribuição defensável neste semestre é uma avaliação delimitada, reproduzível e crítica do Mamba-3 para identificação e previsão, seguida de uma investigação de integração com MPC. Não afirmar ineditismo sem revisão específica da literatura.

### Primeiras ações práticas

- [ ] Registrar o que efetivamente mudou desde 14/09.
- [ ] Discutir com o orientador a planta proposta e a prioridade de identificação nesta fase.
- [ ] Identificar a GPU e validar passagem direta e gradientes do Mamba-3.
- [ ] Congelar tarefa, sinais disponíveis, divisão dos dados e métricas.
- [ ] Gerar dados e produzir a primeira figura de força e posição.
- [ ] Ajustar persistência e ARX antes de iniciar treinamento da rede.
- [ ] Executar o experimento principal e fechar a tabela antes de acrescentar extensões.
