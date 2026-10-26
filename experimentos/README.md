# Base experimental do TCC

O primeiro experimento já foi executado localmente, em CPU, em 12/10/2026. O ambiente isolado está em `.venv/`; não foram instaladas bibliotecas no Python global. Esta etapa contém a planta simulada e as referências persistência e ARX. Mamba-3 e MPC permanecem pendentes.

## Ver o que já existe

- [Resultados e interpretação](../resultados/linear-v1/resumo.md).
- [Código do experimento](experimento_linear.py).
- [Configuração editável](config-linear.json).
- [Registro da fase para apoiar o relatório](../planejamento/registro-experimento-2026-10-12.md).

## Executar

No terminal, a partir da pasta do TCC:

```bash
.venv/bin/python experimentos/experimento_linear.py --output resultados/linear-v2
```

Cada execução precisa de uma pasta de saída nova. O programa se recusa a sobrescrever uma pasta existente, para preservar resultados anteriores. Ele gera os dados, ajusta as referências, escolhe a ordem ARX na validação, avalia o teste e produz quatro figuras em PNG e SVG.

Verificações:

```bash
.venv/bin/python -m unittest discover -s experimentos -p 'test_*.py' -v
```

Foram verificadas a discretização contra integração RK4 independente, a condição de equilíbrio físico, a identificação e simulação de uma trajetória não usada no ajuste, a realimentação das previsões, a rejeição de medições futuras, a causalidade da entrada, a normalização apenas pelo treino e a reprodutibilidade dos dados.

Para reconstruir o ambiente em outra máquina com Python 3.12, caso `.venv` ainda não exista:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experimentos/requirements.txt
```

As versões correspondem à execução registrada. Elas não incluem PyTorch ou os kernels do Mamba-3.

## Contrato dos dados e avaliação

`dados.npz` contém, para cada conjunto `treino`, `validacao` e `teste`:

- `*_u`: forças, com formato `(trajetorias, 2048)`.
- `*_y`: posições medidas, com formato `(trajetorias, 2049)`.
- `*_ids`: identificadores das trajetórias.

O número adicional de saídas permite armazenar a condição inicial e a saída produzida pela última entrada. No instante `t`, são conhecidos `y[t]` e a entrada aplicada `u[t]`; o alvo é `y[t+1]`. O início de cada trajetória é independente. A separação por trajetória ocorre antes de qualquer extração de janelas.

O preditor ARX usa os últimos valores de saída e entrada e um intercepto. Os parâmetros físicos são usados somente na simulação: `fit_arx` não os recebe. As ordens são escolhidas pelo RMSE da trajetória de 50 passos na validação. Configurações cujo erro esteja dentro de uma tolerância absoluta de `1e-12` m do melhor resultado são desempatadas pelo menor número de parâmetros; essa regra evita interpretar diferenças de precisão numérica como ganhos úteis no cenário ideal.

As origens de previsão são as mesmas para todos os modelos, começando após 256 amostras, espaçadas de 16 amostras, com espaço para completar 50 passos. O primeiro passo usa medições reais até a origem; os seguintes usam saídas previstas. Os horizontes 1, 10 e 50 compartilham essas origens.

Os campos `rmse_endpoint_m` e `rmse_trajectory_m` são diferentes: o primeiro mede apenas o ponto final do horizonte; o segundo reúne todos os passos até ele. O denominador de NRMSE é o desvio-padrão dos alvos avaliados no conjunto correspondente, comum aos horizontes. Ele serve para expressar a métrica depois das escolhas, não para normalizar entradas nem selecionar modelos no teste.

Os estados físicos são salvos separadamente em `diagnostico/estados-fisicos.npz`. Eles não entram no ajuste nem na avaliação dos preditores. A normalização disponível para a futura rede usa somente o conjunto de treino. O ARX desta execução é ajustado em unidades físicas.

## Como ler o primeiro resultado

O ARX identifica uma planta linear de segunda ordem, sem ruído e gerada por discretização exata. Um erro próximo da precisão numérica é esperado nessas condições. Isso confirma o funcionamento desta referência no cenário ideal; não representa validação de Mamba-3 ou conclusão sobre sistemas reais.

Os gráficos de um passo podem mostrar curvas quase sobrepostas. A figura de vários passos evidencia por que avaliar somente um passo pode esconder a perda de qualidade durante uma simulação. As figuras mostram trechos predefinidos da primeira trajetória de teste; a tabela usa todas as origens do teste.

O tempo registrado é apenas o ajuste do ARX em CPU. Não é um benchmark de inferência nem uma comparação com GPU.

## Mamba-3 em PyTorch puro

Desde 17/10/2026 o repositório tem um bloco Mamba-3 SISO que roda na CPU e na GPU do Mac (MPS), sem Triton nem CUDA: [mamba3_referencia.py](mamba3_referencia.py). Ele usa as funções de referência dos autores (`mamba3_siso_fwd_ref` e `mamba3_siso_step_ref`), copiadas sem alteração do commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` de [state-spaces/mamba](https://github.com/state-spaces/mamba), licença Apache-2.0. O resto do módulo foi reescrito a partir de `mamba_ssm/modules/mamba3.py`, com os mesmos nomes e formatos de parâmetros.

- `forward(u)` processa a sequência inteira e é usado no treino.
- `step(u, estado)` avança uma amostra por vez e é usado na simulação de vários passos.
- `forward(u, return_state=True)` devolve o estado final, no formato que `step` aceita. Assim é possível aquecer com as medições e seguir realimentando previsões.

Instalar as dependências adicionais:

```bash
.venv/bin/python -m pip install -r experimentos/requirements-mamba3.txt
```

Verificações (na CPU; para a GPU do Mac, prefixar com `TCC_DEVICE=mps`):

```bash
.venv/bin/python -m unittest discover -s experimentos -p 'test_mamba3*.py' -v
```

Execução registrada, com versões, erros e tempos:

```bash
.venv/bin/python experimentos/verificar_mamba3.py --output resultados/mamba3-verificacao-v2
```

A primeira execução está em [resultados/mamba3-verificacao-v1](../resultados/mamba3-verificacao-v1/resumo.md). Os tempos valem para esta implementação de referência, não para os kernels otimizados dos autores.

Limites conhecidos:

- `fwd_ref` tem custo quadrático no comprimento da sequência. Para janelas de até 2.048 amostras e uma camada pequena, isso não pesa.
- Na saída do instante `t`, `fwd_ref` soma e subtrai um termo que depende de `dt` e `trap` em `t+1`. O resultado é causal, mas a saída em `t` varia na casa do arredondamento quando a entrada em `t+1` muda. `step` é exatamente causal.
- Ainda não foi feita a comparação com o módulo oficial numa GPU NVIDIA, carregando o mesmo `state_dict`. Enquanto isso não for feito, descrever como "reimplementação a partir das funções de referência dos autores".

## Treino e avaliação da rede Mamba-3 (E002)

[experimento_mamba3.py](experimento_mamba3.py) treina a rede nos dados do E001 e a avalia no mesmo protocolo. A configuração fica em [config-mamba3.json](config-mamba3.json).

```bash
.venv/bin/python experimentos/experimento_mamba3.py --output resultados/mamba3-v2
```

Por padrão usa a GPU do Mac (`--device mps`); `--device cpu` também funciona. `--smoke` roda duas épocas para testar o fluxo. `--validation-only` para depois da escolha, sem ler o teste, e serve para ajustar o treino durante o desenvolvimento.

O que o script faz, em ordem:

1. Lê `resultados/linear-v1/`: dados, normalização do treino, origens e horizontes do E001.
2. **Rede:** Linear(2→64), um bloco Mamba-3 com RMSNorm e conexão residual, Linear(64→1). No instante `t` entram `u[t]` e `y[t]` normalizados; sai `y[t+1]`.
3. **Treino:** erro quadrático de um passo, em janelas de 512 amostras das trajetórias de treino, com passo de 128 entre janelas. O estado começa do zero em cada janela, e as primeiras 128 posições ficam fora da perda. AdamW, taxa com decaimento em cosseno, recorte de gradiente em 1,0, até 300 épocas.
4. **Escolha na validação:** a cada 10 épocas, simula 50 passos nas 545 origens da validação e guarda o melhor checkpoint. Cada taxa de aprendizado candidata roda com três sementes. Vence a de menor média, entre as sementes, do RMSE da trajetória de 50 passos.
5. **Teste:** só depois da escolha. Persistência e ARX são recalculados a partir de `modelo-arx.json` e conferidos contra as métricas gravadas no E001. A rede é avaliada nas mesmas 545 origens; o script confere que os alvos são idênticos aos do ARX.

Em cada origem `t`, a rede é aquecida com as 256 medições de `t-255` a `t`, com `forward`, e depois continua com `step`, realimentando as próprias previsões. Saídas reais posteriores à origem não são lidas; um teste substitui-as por `NaN` e confere que as previsões não mudam.

Verificações do protocolo: [test_experimento_mamba3.py](test_experimento_mamba3.py) confere o alinhamento das janelas, as origens e os alvos, a ausência de medições futuras e a continuação do estado. Esta última é comparada com recalcular a sequência inteira a cada passo.

`experimento_linear.py` ganhou duas funções, `evaluation_origins` e `summarize_rollouts`, para que a rede use o mesmo cálculo de origens e métricas do ARX. Executado de novo depois dessa mudança, o E001 reproduziu as mesmas métricas e os mesmos dados; só os tempos de ajuste mudaram.

## Planta de Duffing e NARX polinomial (E003 e E004)

[experimento_duffing.py](experimento_duffing.py) troca a planta pelo oscilador de Duffing, `m p'' + c p' + k p + k3 p³ = u`, integrado por RK4 com subpassos sob segurador de ordem zero. A força passa a ter amplitude sorteada em cada patamar (APRBS), para excitar a não linearidade em vários níveis. A divisão por trajetória, as origens, os horizontes e as métricas são os do E001.

Além de persistência e ARX, o script ajusta um **NARX polinomial**: todos os monômios até o grau `g` de `y[t..t-n+1]` e `u[t..t-n+1]`, por mínimos quadrados do erro de um passo, com `n` e `g` escolhidos pela simulação de 50 passos na validação. Candidatos que divergem na simulação são registrados e descartados.

```bash
.venv/bin/python experimentos/experimento_duffing.py --output resultados/duffing-v2
.venv/bin/python experimentos/experimento_duffing.py --config experimentos/config-duffing-ruido.json --output resultados/duffing-ruido-v2
```

Com ruído de medição, o script também grava a posição sem ruído em `diagnostico/saida-sem-ruido.npz`, fora dos dados dos modelos. As mesmas previsões são então medidas contra a posição sem ruído, em `metricas-teste-sem-ruido.csv`. Isso serve só para relatar o erro; nenhuma escolha usa esses valores.

A rede Mamba-3 usa o mesmo [experimento_mamba3.py](experimento_mamba3.py), apontando para a execução de referência:

```bash
.venv/bin/python experimentos/experimento_mamba3.py --config experimentos/config-mamba3-duffing.json --output resultados/mamba3-duffing-v2
.venv/bin/python experimentos/experimento_mamba3.py --config experimentos/config-mamba3-duffing-ruido.json --output resultados/mamba3-duffing-ruido-v2
```

Quando a execução de referência tem `modelo-narx.json`, o NARX entra nas tabelas e nas figuras ao lado de persistência e ARX. As verificações estão em [test_experimento_duffing.py](test_experimento_duffing.py): sem o termo cúbico, a integração coincide com a discretização exata do E001; o equilíbrio estático resolve a equação cúbica; os subpassos convergem; e o NARX recupera exatamente um mapa polinomial conhecido.

[figuras_relatorio_2.py](figuras_relatorio_2.py) recarrega os modelos gravados do E003 e do E004 e produz as duas figuras de comparação do relatório 2 em `relatorio-2/figuras/`. Não treina nem escolhe nada.

[diagnostico_e004.py](diagnostico_e004.py) é um diagnóstico **posterior ao teste**: amplia a grade de ARX e NARX nos dados do E003/E004 e separa as fontes de erro do NARX com ruído (coeficientes enviesados contra condição inicial ruidosa). Serve para interpretar os resultados e não substitui as execuções registradas. Saída em `resultados/diagnostico-e004-v1/`.
