# Revisão do plano para o relatório 2 - 17/10/2026

**Situação:** faltam nove dias para a entrega de **26/10, segunda-feira, até 20h**. A apresentação remota é em **27/10**.
**Base:** [plano de 11/10](plano-relatorio-2.md), [registro de 12/10](registro-experimento-2026-10-12.md), vault do grupo e relatório 1.
**Natureza:** proposta de planejamento. Nada aqui foi combinado com o grupo ou com o orientador.
**Atualização no mesmo dia:** o bloco Mamba-3 em PyTorch foi montado e verificado no Mac (seção 3, item 1). Veja `experimentos/README.md` e `resultados/mamba3-verificacao-v1/`. Ainda em 17/10, a rede foi treinada e avaliada no protocolo do E001 (`resultados/mamba3-v1/`): em 50 passos, (1,16 ± 0,10) × 10⁻² m, contra 0,684 m da persistência. A entrega mínima técnica está adiantada; o que falta é revisão do grupo, orientador e redação.

## 1. Onde estamos

O grupo não avançou desde o relatório 1 (14/09). O único material novo é a base de 11 e 12/10: o plano, o experimento E001 com persistência e ARX, e o vault. Ela foi preparada com apoio do assistente e ainda não foi revisada pelos integrantes.

Comparado ao plano de 11/10:

| Item | Previsto | Situação em 17/10 |
|---|---|---|
| Recorte discutido com orientador | 11 a 13/10 | Não feito |
| Responsabilidades combinadas | 11 a 13/10 | Não feito |
| Ambiente do Mamba-3 validado | até 14/10 | Não feito |
| Dados, ARX e primeiros gráficos | 14 a 16/10 | Adiantado, feito em 12/10 |
| Treinamento da rede pequena | 17 a 20/10 | Começa agora |

O que falta para a entrega mínima se concentra em uma frente: **treinar e avaliar uma rede com Mamba-3 no protocolo do E001**. O resto é revisão e redação.

## 2. Mamba-3 e HuggingFace

**Este experimento não precisa do HuggingFace.** Situação conferida em 17/10/2026:

- No Hub existem os checkpoints `state-spaces/mamba3-siso-*` e `mamba3-mimo-*`, de 187M a 1,5B parâmetros, publicados em 27/07/2026. São modelos de linguagem, treinados em texto (FineWeb-Edu) com tokenizador do Llama 3.1. Servem para gerar texto, não para prever a posição de uma massa.
- A biblioteca `transformers` (v5.20.1, 16/10/2026) tem apenas Mamba e Mamba-2. Não há classe Mamba-3; o pedido para incluí-la foi fechado.
- O TCC treina do zero uma rede de dezenas de milhares de parâmetros sobre dois sinais numéricos. O código do bloco está no pacote `mamba_ssm`, do repositório [state-spaces/mamba](https://github.com/state-spaces/mamba), e não no HuggingFace.

O pacote `mamba_ssm` traz o Mamba-3 desde a versão 2.3.2 (PyPI, 09/05/2026). O `main` está nove commits à frente (último em 22/07/2026), e um deles muda a ativação de A ([#962](https://github.com/state-spaces/mamba/pull/962)). Fixar o commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` e registrá-lo nos resultados.

**O que estudar é PyTorch:** tensores, autograd, `nn.Module`, otimizador e laço de treino. O tutorial [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) cobre isso em uma tarde. Depois, ler o script de treino do grupo.

## 3. Caminho técnico recomendado

O que roda em cada máquina, segundo o código e as issues do repositório oficial:

| Caminho | MacBook Air M5 | PC com RTX 4060 | Colab gratuito (T4) |
|---|---|---|---|
| Kernels oficiais, treino SISO (Triton) | Não: Triton só existe para Linux com NVIDIA | Relatado funcionando, via Linux ou WSL2 ([#1017](https://github.com/state-spaces/mamba/issues/1017)) | Sem relato; provável falha (bf16 e Triton em GPUs Turing) |
| `step` oficial, inferência passo a passo | Não | Documentado só em H100; bugs abertos ([#1024](https://github.com/state-spaces/mamba/issues/1024), [#1014](https://github.com/state-spaces/mamba/issues/1014)) | Não |
| MIMO oficial (TileLang) | Não | Provável falha por memória compartilhada ([#917](https://github.com/state-spaces/mamba/issues/917), [#1044](https://github.com/state-spaces/mamba/issues/1044)) | Não |
| Referência em PyTorch dos autores | Sim, CPU ou MPS | Sim | Sim |

A referência em PyTorch são as funções `mamba3_siso_fwd_ref` (sequência inteira, diferenciável) e `mamba3_siso_step_ref` (uma amostra por vez, com estado). Elas ficam em `tests/ops/triton/test_mamba3_siso.py` e não são instaladas com o pacote. O módulo `Mamba3` depende de um RMSNorm em Triton e não roda fora de GPU NVIDIA.

### Recomendação

1. **Caminho principal: Mamba-3 SISO em PyTorch puro, no Mac.** Copiar as duas funções de referência do commit fixado e reescrever em PyTorch as projeções do módulo `Mamba3` (cerca de 500 linhas, das quais só o caminho SISO interessa). A rede usa `fwd_ref` no treino e `step_ref` na simulação de vários passos.
   - Roda já, em fp32, sem depender do PC.
   - Oferece simulação recorrente com estado persistente, que os kernels oficiais não entregam de forma confiável no hardware disponível.
   - O repositório já tem um teste que compara `step_ref` com `fwd_ref`; repetir essa verificação no nosso código.
   - Medida neste Mac em 17/10: um scan seletivo ingênuo em PyTorch, com batch 32 × 512 amostras e as dimensões do plano, levou cerca de 0,27 s por passo de treino, igual na CPU e no MPS. Com os dados do E001, isso dá poucos minutos por treinamento.
   - Limite: os tempos não representam os kernels otimizados e não servem como benchmark de latência.
2. **Verificação no PC, se der tempo.** Instalar WSL2 com Ubuntu, PyTorch com CUDA e `mamba_ssm` no mesmo commit. Passar a mesma entrada pelo módulo oficial e pelo nosso, com os mesmos pesos, e comparar as saídas. Com essa verificação, o relatório pode afirmar fidelidade ao código oficial. Sem ela, descrever como "reimplementação a partir das funções de referência dos autores, não comparada com os kernels".
3. **Se a montagem travar até 18/10:** usar [VikramLex/mamba3-minimal](https://github.com/VikramLex/mamba3-minimal) (PyTorch, Apache-2.0), declarando os desvios conhecidos: A não depende da entrada e os ângulos não são limitados. Último recurso: GRU, sem chamá-la de Mamba-3.
4. **Não usar o Colab gratuito** para os kernels oficiais.

Configuração do plano: `d_model=64, d_state=64, expand=2, headdim=64` resulta em duas cabeças. A rede precisa de uma projeção linear de 2 para 64 na entrada e de 64 para 1 na saída.

### Cuidados se usar os kernels oficiais no PC

- Instalar o PyTorch com CUDA antes; relatos usam torch 2.9 ou 2.10 com CUDA 12.8. Depois:
  `pip install "git+https://github.com/state-spaces/mamba.git@e9594ce1c732d97440f0332fdc43170a2294dbfa" --no-build-isolation`
- `pip install apache-tvm-ffi==0.1.9`, porque a 0.1.12 quebra o `import` ([#986](https://github.com/state-spaces/mamba/issues/986)).
- Usar comprimento de sequência múltiplo de 4, de preferência de 64 ([#886](https://github.com/state-spaces/mamba/issues/886)). Atenção ao deslocar o alvo para \(y_{t+1}\): 511 ou 2047 quebram o backward.
- Não usar `step()`.
- O kernel converte os cálculos internos para bf16. Há relato, ainda sem resposta, de 3 a 7,5% de erro relativo contra a referência numa RTX 4060 (#1017). Num problema em que o ARX chega a \(10^{-13}\) m, isso pesa e precisa ser declarado.

### Mamba-2, se entrar

Comparar pelo mesmo tipo de caminho: as duas referências em PyTorch, ou os dois kernels oficiais. Para o Mac, a função `ssd_minimal_discrete` de `mamba_ssm/modules/ssd_minimal.py` pode ser copiada sem Triton. Misturar caminhos confunde arquitetura com precisão numérica.

### Adaptação da avaliação

`rollout` e `evaluate`, em `experimentos/experimento_linear.py`, chamam o modelo origem por origem com todo o histórico. Para a rede, processar as 545 origens em lote: aquecer o estado com as 256 medições anteriores a cada origem e seguir 50 passos com `step_ref`, realimentando as previsões. As origens, os horizontes e as métricas devem ser exatamente os do E001.

## 4. Cronograma até a entrega

| Dia | Trabalho | Evidência ao terminar |
|---|---|---|
| Sáb 17 e dom 18/10 | Montar o Mamba-3 em PyTorch no Mac (seção 3); verificar passagem direta, gradientes e `step_ref` contra `fwd_ref`. Grupo lê E001, Protocolo comum e Previsão e simulação. Quem tiver acesso ao PC verifica o sistema operacional e começa o WSL2 | Log da primeira execução com versões e commit; dúvidas do grupo anotadas |
| Seg 19/10 | Conversa com o orientador sobre o recorte. Reunião do grupo para dividir as frentes | Nota de reunião em `vault/`; decisões registradas |
| Seg 19 a qua 21/10 | Script de treino: janelas, normalização do treino, perda de um passo, validação. Avaliação com aquecimento e realimentação nas mesmas origens do E001 | Curvas de treino/validação; primeira tabela na validação |
| Qua 21/10 | **Ponto de decisão:** a rede aprende? | Ver seção 6 |
| Qui 22/10 | Congelar configuração; avaliar no teste; repetir com três sementes | `resultados/mamba3-v1/` com métricas, figuras e registro |
| Qui 22 e sex 23/10 | Extensão, se o principal estiver fechado (seção 7) | Uma rodada extra, registrada à parte |
| Sex 23 e sáb 24/10 | Redigir o relatório e montar os slides | Primeira versão completa |
| Dom 25/10 | Revisão do grupo; enviar ao orientador | Versão revisada |
| Seg 26/10 | Conferir PDF e enviar no Moodle e ao orientador até 20h | Comprovante de envio |
| Ter 27/10 | Apresentar (10 min + 5 min de perguntas); pedir ao orientador o envio da nota à banca | Feedback registrado no vault |

A redação pode começar antes: as seções de estágio anterior, fundamentação e metodologia não dependem do resultado da rede.

## 5. Divisão proposta

| Frente | O que faz | Entrega para o grupo |
|---|---|---|
| A. Modelo | Ambiente, rede Mamba-3, treino e avaliação | Checkpoint, métricas e diagnóstico técnico |
| B. Dados e comparação | Revisar simulador, dados e ARX; preparar Mamba-2 ou a extensão escolhida | Revisão do E001; segunda comparação, se houver tempo |
| C. Relatório e apresentação | Estágio anterior, fundamentação, metodologia, figuras finais e slides | Relatório integrado e roteiro de 10 minutos |

Vinicius tem interesse em testar o Mamba-3, o que sugere a frente A. A atribuição das outras frentes fica para a reunião do grupo.

Todos precisam saber explicar o alinhamento temporal (entra \(u_t, y_t\), sai \(y_{t+1}\)), a divisão por trajetória e a diferença entre previsão de um passo e simulação de vários passos. A banca vai perguntar.

## 6. Pontos de decisão

- **Dom 18/10 à noite:** sem passagem direta e gradientes do Mamba-3 em nenhum caminho, seguir com a alternativa da seção 3 e registrar o bloqueio com evidência.
- **Qua 21/10:** se o erro de um passo na validação não ficar abaixo do da persistência, parar de acrescentar coisas e diagnosticar: alinhamento dos índices, normalização, taxa de aprendizado, gradientes.
- **Qui 22/10:** congelar escolhas antes de olhar o teste. A partir daqui, nada de nova planta, MIMO ou controlador.

## 7. Se sobrar tempo, nesta ordem

1. Três sementes de treinamento, com dispersão.
2. Mamba-2 com a mesma estrutura externa, contando os parâmetros reais.
3. Uma rodada com ruído de medição, ou a mola cúbica.

Na planta linear sem ruído, o ARX chega à precisão numérica (cerca de \(10^{-13}\) m em 50 passos). **A rede não vai superá-lo, e não precisa.** O resultado defensável para o relatório 2 é: a rede aprende a dinâmica, com erro X em 1, 10 e 50 passos, contra a referência ideal. A comparação em que a rede pode mostrar alguma vantagem é a planta não linear ou com ruído, que fica naturalmente para novembro.

## 8. Levar ao orientador em 19/10

1. O massa-mola-amortecedor linear serve como primeiro caso de verificação?
2. A entrega 2 pode priorizar identificação e previsão, deixando MPC para novembro?
3. Qual extensão vale mais para o semestre: ruído, não linearidade ou Mamba-2?
4. Uma implementação de referência em PyTorch, verificada contra o código oficial, é aceitável no lugar dos kernels oficiais?
5. Que nível de evidência de controle é esperado em dezembro?
