# Levantamento do Silverbox (E007)

Data: 21/11/2026. Só leitura: nenhum dado foi baixado, nada foi instalado. Preparado para a sessão que vai implementar `dados_silverbox.py` e `config-silverbox.json` (plano A3, Onda 4, bloco 5).

**Convenções.** "Visto" = a página ou o PDF foi aberto e lido nesta sessão (os PDFs foram baixados para a pasta temporária da sessão só para leitura; o conjunto de dados não foi). "Derivado" = cálculo meu a partir de algo visto. "Não verificado" = só apareceu em resumo de busca, ou não apareceu. Todos os erros abaixo são **RMSE de simulação livre em mV**, salvo indicação.

---

## 1. Fonte oficial e citação

| Item | Resultado | Fonte |
|---|---|---|
| Página oficial | Google Sites do "Nonlinear Benchmark", seção Silverbox | https://sites.google.com/view/nonlinear-benchmark/benchmarks/silverbox (visto) |
| Arquivo de dados | `SilverboxFiles.zip` (título da página de visualização do Drive), no Google Drive | https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view?usp=sharing (página de visualização vista; o arquivo não foi baixado) |
| Tamanho do zip | **5.793.999 bytes** (~5,8 MB). Esse número está no código do carregador oficial (`download_size=5793999`); **não confirmei no Drive** | https://raw.githubusercontent.com/MaartenSchoukens/nonlinear_benchmarks/HEAD/nonlinear_benchmarks/benchmarks.py (visto, função `Silverbox`) |
| Conteúdo do zip | Pasta `SilverboxFiles/` com `SNLS80mV.mat` (registro principal, treino e teste) e `Schroeder80mV.mat` (registro extra de multisseno de fase Schroeder). A página diz que o zip traz os dados em `.mat`, uma versão `.csv` dos mesmos dados e o registro extra. Os nomes dos arquivos vêm do código do carregador; **tamanhos individuais não verificados** | página oficial e `benchmarks.py` (vistos) |
| Variáveis nos `.mat` | `V1` = entrada, `V2` = saída medida | página oficial e nota técnica (vistas) |
| Nota técnica | PDF de 6 páginas (375.681 bytes) linkado como "technical note": https://drive.google.com/file/d/1EgQGYn1r949VChEhUTRkOe5xToxiFszE/view?usp=sharing. O título extraído do PDF é "Data for benchmarking in nonlinear system identification" (Wigren e Schoukens), que parece ser o preprint do artigo do ECC; o título do ECC é outro (abaixo) | visto |
| Planilha de resultados | Google Sheets embutida na página: https://docs.google.com/spreadsheets/d/11qXEYX8NhD5URPkoXOv8x_8qXkcHzSuMJBs3Zf8pppQ (exportei como CSV para ler) | vista |
| Carregador oficial | Pacote `nonlinear-benchmarks` 1.0.1 (PyPI), repositório https://github.com/MaartenSchoukens/nonlinear_benchmarks (nome devolvido pela API do GitHub; o artigo [C] cita o mesmo repositório como GerbenBeintema/nonlinear_benchmarks, provavelmente redirecionado). Dependências dele: numpy, tqdm, progressbar, scipy, requests, rarfile | https://pypi.org/project/nonlinear-benchmarks/ e `pyproject.toml` (vistos) |
| Licença / condições | A página oficial **não declara licença para os dados**. O artigo diz que os conjuntos estão "disponíveis para download gratuito" (resumo da nota técnica). O código do carregador é BSD 3-Clause (arquivo `LICENSE`, visto). A página pede citar a referência abaixo | páginas citadas |
| Histórico do link | O carregador traz, comentado, o endereço antigo `http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip`; o atual é o do Drive | `benchmarks.py` (visto) |

**Referência a citar** (texto da própria página oficial, seção "Cite"):

> T. Wigren and J. Schoukens. Three free data sets for development and benchmarking in nonlinear system identification. 2013 European Control Conference (ECC), pp. 2933-2938, July 17-19, 2013, Zurich, Switzerland, doi: 10.23919/ECC.2013.6669201.

O relatório 2 cita "Wigren e Schoukens, ECC 2013": está correto. O sistema em si é descrito em R. Pintelon e J. Schoukens, *System Identification: A Frequency Domain Approach*, IEEE Press, 2001 (referência [11] da nota técnica). A página também agradece a Johan Schoukens (criação) e a Torbjörn Wigren (hospedagem).

---

## 2. Conteúdo do conjunto

Fonte: nota técnica (seções V e VI) e carregador oficial, ambos vistos.

- **Sistema:** circuito eletrônico que implementa um oscilador de Duffing: LTI de 2ª ordem com não linearidade estática cúbica em realimentação. Equações da nota: m·y'' + d·y' + k(y)·y = u, com k(y) = a + b·y². O circuito é "próximo, mas não exatamente igual" ao modelo ideal; a relação sinal-ruído é alta (a nota modela sem ruído de medição).
- **Variáveis e unidades:** `V1` = entrada u (força, em volts); `V2` = saída y (deslocamento, em volts). As figuras da nota têm eixo em V; o carregador multiplica o RMSE por 1000 para reportar em mV.
- **Taxa de amostragem:** fs = 10⁷/2¹⁴ ≈ **610,35 Hz** (nota técnica, eq. 12; o carregador usa `sampling_time = 1/610.35`). Derivado: 50 passos ≈ 82 ms.
- **Excitação (nota, seção VI-C):** o sinal de referência tem duas partes.
  1. **40.000 amostras** de ruído gaussiano branco filtrado por Butterworth discreto de 9ª ordem, corte em 200 Hz, com amplitude variando linearmente de zero ao máximo. É o trecho "arrow" (flecha) de amplitude crescente.
  2. **10 realizações sucessivas de um multisseno ímpar aleatório**, período 1/f0 com f0 = fs/8192 (ou seja, 8192 amostras por realização), 1342 harmônicos ímpares até 200 Hz, fases independentes uniformes. A nota diz que as realizações foram separadas por 100 zeros; os índices do carregador (seção 3) mostram espaçamento de 8692 amostras entre inícios, isto é, 500 amostras de folga. **Divergência não resolvida**: conferir ao abrir o arquivo.
- **Número total de amostras:** **não verificado** (não abri o arquivo). Os índices do carregador chegam a 127.400, e os artigos falam em 40.000 (arrow) + 86.750 (multisseno) (Floren et al., arXiv 2602.22964, seção V-A, visto). Portanto o registro tem ao menos 127.400 amostras.
- **Registro extra:** `Schroeder80mV.mat`, multisseno de fase Schroeder, não usado na divisão oficial. Não há razão para usá-lo no E007.

---

## 3. Divisão treino/validação/teste e métrica padrão

### 3.1 Divisão oficial (carregador `nonlinear_benchmarks.Silverbox()`)

Índices em `SNLS80mV.mat`, base 0, intervalo `[a:b)` (código visto em `benchmarks.py`; os comprimentos são derivados):

| Conjunto | Índices | Amostras | Observação |
|---|---|---|---|
| Teste "arrow full" | `[100:40575]` | 40.475 | trecho de amplitude crescente; vai além da amplitude vista no treino |
| Teste "arrow no extrapolation" | `[100:32100]` (primeiras 32.000 de arrow full) | 32.000 | subconjunto de arrow full, na faixa de amplitude do treino (README: "é um subconjunto de `test_arrow_full`") |
| Região de multisseno usada | `[40650:127400]` | 86.750 | |
| Treino+validação (`train_val`) | `[40650:105712]` | 65.062 (75%) | corte `int(86750*0.75)` |
| Teste multisseno | `[105712:127400]` | 21.688 (25%) | |

- Retorno: `train_val, test = Silverbox()`; `test_multisine, test_arrow_full, test_arrow_no_extrapolation = test`.
- **Janela de inicialização de estado: 50 amostras** em todos os três testes (`state_initialization_window_length = 50`, README e código). O modelo pode usar `u` inteiro e só `y[:50]`.
- **Não há validação oficial**: o `train_val` deve ser subdividido pelo usuário (o template de submissão diz: "você pode dividir o treino em treino e validação... não use o teste para nenhuma decisão"). Fonte: `submission_examples/silverbox.py` (visto).
- Derivado dos índices das 10 realizações do carregador (início k = 40986, 49678, 58370, 67062, 75754, 84446, 93138, 101830, 110522, 119214, 8192 amostras cada, usado em `return_multisine_realizations`): o treino cobre as realizações 1 a 7 e 3.882 amostras da 8; o teste multisseno cobre o resto da 8 (4.310 amostras) mais a 9 e a 10 (a 10 perde as últimas 6 amostras). O teste multisseno é, portanto, de realizações de fase diferentes das do treino, mas **a realização 8 é cortada ao meio entre treino e teste**.
- Treino e teste são de natureza diferente: o treino é só multisseno de amplitude constante; o teste "arrow" é ruído filtrado de amplitude crescente, que passa da amplitude do treino. O README diz que o teste sem extrapolação é a parte da flecha dentro da faixa do treino.

### 3.2 Métrica oficial

- **RMSE em mV, simulação livre** (cabeçalho da planilha oficial: "Test Results (RMSE) Free-Run Simulation"; template: `1000*RMSE(y[n:], y_modelo[n:])`, com n = 50). Reporta-se um vetor de três números: `[multisseno; arrow full; arrow no extrap.]` (template, linha "RMSE to submit").
- RMSE = sqrt(mean((y_modelo − y)²)) (`error_metrics.py`, visto). Sem previsão de um passo.
- Também aparecem FIT (%) e NRMSE em alguns artigos; não são a métrica oficial.

### 3.3 Divisões usadas em artigos (não são a oficial)

| Artigo | Divisão descrita | Fonte (visto) |
|---|---|---|
| Beintema et al. (SUBNET), 2021 | Teste: 40.000 amostras de ruído filtrado de amplitude crescente. Treino: 87.000 amostras de multisseno de amplitude constante; o texto diz que as últimas 30.000 monitoram o desempenho e as 21.000 restantes formam a validação (redação ambígua). Observa que a amplitude do treino é menor que a máxima do teste | arXiv 2012.07697, seção 3.2 |
| Bonassi et al. (S4/S5/LRU), IFAC 2024 | Dez experimentos de multisseno de 8192 amostras; validação = 76 subsequências de 512 passos de um experimento sorteado; treino = 684 subsequências dos outros nove. Teste: 40.500 amostras de ruído filtrado com média crescente; reporta "primeiras 25.000 amostras" (interpolação) e "completo" | arXiv 2312.06211, seção 6 |
| Bemporad (NAILM), 2022 | Teste: primeiras 40.000 amostras; as 10 realizações de multisseno viram 10 trechos de ~8.600 amostras de treino, sem os intervalos entre elas | arXiv 2112.15348, seção 5.1.1 |
| Stable MGU (2026) | Usa "a divisão padrão da literatura": treino/validação 80/20 em subsequências de 250 passos; reporta os 3 testes oficiais | arXiv 2603.03017, seção 6.1 |
| Champneys et al. (baselines), 2024 | Carregador oficial; seleção de ordem/hiperparâmetros no conjunto de validação separado do treino | arXiv 2405.10779, seção 2 e 3 |
| DS3M, nov/2026 | Três números, ordem multisseno/arrow full/no extrap., coerente com o carregador oficial; histórico de 50 amostras nos hiperparâmetros | arXiv 2611.25274, Tabela 2 e 6 |

---

## 4. Resultados publicados no teste

Todas as colunas em **mV (RMSE de simulação livre)**. Tripla = `[multisseno; arrow full; arrow sem extrapolação]` (ordem oficial). "—" = não reportado. Resumo de busca sem abertura do artigo não foi incluído.

**Fontes das tabelas (todas abertas nesta sessão):**
- **[P]** Planilha oficial de resultados (CSV exportado): https://docs.google.com/spreadsheets/d/11qXEYX8NhD5URPkoXOv8x_8qXkcHzSuMJBs3Zf8pppQ
- **[C]** Champneys, Beintema, Tóth, Schoukens, Rogers, *Baseline Results for Selected Nonlinear System Identification Benchmarks*, arXiv 2405.10779v2: https://arxiv.org/pdf/2405.10779 (Tabelas 3 e 4, p. 5)
- **[B]** Bonassi, Andersson, Mattsson, Schön, *Structured state-space models are deep Wiener models*, arXiv 2312.06211: https://arxiv.org/pdf/2312.06211 (Tabela 1)
- **[S]** Beintema, Tóth, Schoukens, *Nonlinear state-space identification using deep encoder networks*, arXiv 2012.07697: https://arxiv.org/pdf/2012.07697 (Tabela 2)
- **[N]** Bemporad, arXiv 2112.15348: https://arxiv.org/pdf/2112.15348 (Tabela 3)
- **[M]** Stable Minimal Gated Unit, arXiv 2603.03017: https://arxiv.org/pdf/2603.03017 (Tabela 1)
- **[D]** Roschkowski e Gernandt, DS3M, arXiv 2611.25274: https://arxiv.org/pdf/2611.25274 (Tabela 6)
- **[O]** *Orthogonal Transforms in Neural Networks Amount to Effective Regularization*, arXiv 2305.06344: https://arxiv.org/pdf/2305.06344 (Tabela 3)
- **[K]** SS-KAN, arXiv 2506.16392: https://arxiv.org/pdf/2506.16392 (Tabela I)

### 4.1 Métodos tradicionais

| Método | Tipo | Valor | Fonte e onde |
|---|---|---|---|
| LTI SS (melhor aproximação linear) | tradicional, linear | 6,96 / 14,4 / 6,58 | [C] Tabela 3 |
| ARX linear | tradicional, linear | 6,95 / 14,2 / 6,59 | [C] Tabela 3 |
| ARX linear (versão atualizada) | idem | 6,95 / 14,3 / 6,58 | [P] linha ARX, submissão de out/2025 |
| NARX polinomial (pNARX) | tradicional | 0,640 / 2,25 / 0,571 | [C] Tabela 3 (v2, ago/2024) |
| NARX polinomial (PNARX, atualizado) | tradicional | 0,505 / 1,41 / 0,455 | [P] linha PNARX, "Updated results", out/2025 |
| GP-NARX | tradicional/kernel | 0,301 / 0,60 / 0,259 | [C] Tabela 3 |
| GPNARX (atualizado) | idem | 0,418 / 0,772 / 0,373 | [P] |
| NLARX "ms8c50" do SysID Toolbox (reexecutado) | tradicional (NARX) | 1,05 (arrow, teste de 40.000 amostras); 0,30 no artigo original dele | [N] Tabela 3 |
| PNLSS (Paduart et al., 2010) | tradicional, estado-espaço polinomial | 0,26 (arrow full, divisão pré-oficial) | [C] Tabela 4; [P] linha "Polynomial Nonlinear State-Space", rotulada "Legacy results; Published before official data split". O artigo original **não foi aberto** |
| NL-LFR com inferência e aprendizado (Floren et al., 2025) | tradicional/estruturado (usou conhecimento da cúbica) | 0,289 / 0,334 / 0,257 | [P] (arXiv 2503.14409); repetido em [D] Tabela 6 como 0,29 / 0,33 / 0,26. Treino 1 a 60 s, GPU de laptop |
| Poly-LFR (Van Mulders et al., 2013) | tradicional | 0,35 (legado) | [P] |
| Genetic Programming NARX (Khandelwal) | tradicional, NARX por programação genética | 0,36 (legado) | [P] |
| NL-LFR (Schoukens e Tóth, 2020) | tradicional | 0,38 (legado) | [P] |
| Local linear models (Verdult, 2004) | tradicional | 1,1 e 1,3 (validação e teste, conforme [S]; a planilha traz [1,1; 1,3]) | [S] Tabela 2; [P] |
| Grey-box NARX (Ljung et al., 2004) | tradicional | 0,3 (completo) | [B] Tabela 1 |
| Wiener (Tiels, 2015) | tradicional, bloco | 9,2 (completo); 1,9 (primeiras 25.000) | [B] Tabela 1, citando Tiels |
| Melhor aproximação linear (Marconato et al., 2012; Tiels) | linear | 13,5 [S]; 13,7 [B] | [S] Tabela 2; [B] Tabela 1 |
| PNLSS em volts | | 0,0003 V = 0,3 mV | [K] Tabela I (conjunto de teste não especificado) |

**NARMAX:** não achei resultado rotulado NARMAX no Silverbox nas fontes abertas. **Hammerstein-Wiener:** idem; só o modelo de Wiener de Tiels acima. Marcar como "não verificado / não encontrado".

### 4.2 Métodos neurais e SSMs

| Método | Tipo | Valor | Fonte e onde |
|---|---|---|---|
| SUBNET discreto (encoder de estado) | neural, estado-espaço | 0,36 / 1,4 / 0,32 | [C] Tabela 4; [P]; [S] Tabela 2; [D] Tabela 6. Treino de 1 a 4 dias em CPU de laptop ([P]) |
| DS3M (nov/2026) | neural, estado-espaço profundo estruturado, kernels Triton | 0,23 / 0,74 / 0,23 | [D] Tabela 6, treino de 0,5 h |
| dynoNet | neural/híbrido | 1,218 / 5,609 / 1,348 | [P] (arquitetura única para todos os benchmarks, sem ajuste, nota do autor) |
| RNN simples | neural | 1,64 / 4,88 / 1,73 | [C] Tabela 3 |
| GRU | neural | 1,49 / 2,80 / 0,947 | [C] Tabela 3 |
| LSTM | neural | 1,50 / 2,68 / 0,960 | [C] Tabela 3 |
| OLSTM (LSTM ortogonal) | neural | 1,51 / 2,38 / 1,08 | [C] Tabela 3 |
| MLP FIR | neural | 18,9 / 20,2 / 15,3 | [C] Tabela 3 |
| MLP NARX | neural | 2,80 / 5,18 / 2,40 | [C] Tabela 3 |
| Mesmas redes, versão de out/2025 | | RNN 1,95/3,5/1,54; GRU 1,07/4,37/1,17; LSTM 1,05/6,13/1,35; OLSTM 1,04/5,87/1,27; MLPNARX 1,35/3,06/1,32; MLPFIR 18,1/21,6/16,9 | [P] |
| S4 | SSM profundo | 0,81 (primeiras 25.000) / 4,73 (completo) | [B] Tabela 1 |
| S5 | SSM profundo | 0,73 / 3,56 | [B] Tabela 1 |
| S5R (autovalores aleatórios) | SSM profundo | 0,77 / 4,58 | [B] Tabela 1 |
| LRU | SSM profundo | 0,73 / 4,18 | [B] Tabela 1 |
| LSTM (Andersson et al., 2019) | neural | 0,31 (primeiras 25.000) / 4,0 (completo) | [B] Tabela 1, citando Andersson 2019 |
| TCNN (Andersson et al., 2019) | neural | 0,75 / 4,9 | [B] Tabela 1 |
| LSTM recorrente ([17] de Bemporad) | neural | 2,20 (arrow, 40.000 amostras) | [N] Tabela 3 |
| NAILM (Bemporad) | neural (RNN com treino de mínimos quadrados) | 0,35 | [N] Tabela 3 |
| State-space encoder, nx=4 e nx=2 (versões da primeira publicação) | neural | nx=4: val 0,36, teste 1,4 (0,32 sem extrapolação); nx=2: val 1,0, teste 2,4 (0,83) | [S] Tabela 2 |
| MGU estável (MGUWS) | neural com garantia de estabilidade | 5,27 / 6,88 / 4,37 | [M] Tabela 1 |
| GRU estável (GRULA) | neural com garantia de estabilidade | 24,67 / 24,38 / 19,37 | [M] Tabela 1 |
| FNN, MLP, FSNN | neural | 4,1; 3,9; 2,3 (teste, divisão não detalhada) | [O] Tabela 3 |
| SS-KAN | neural, grey-box | 0,0039 V = 3,9 mV (teste não especificado) | [K] Tabela I |

### 4.3 Mamba ou outra SSM nesse benchmark

- **Mamba: não encontrei nenhum resultado no Silverbox.** Busquei por "Mamba" + Silverbox/system identification/benchmarks de três formas; os PDFs abertos (S4/S5 [B], DS3M [D], MGU [M], baselines [C], SUBNET [S]) não trazem resultados de Mamba no Silverbox. O DS3M só menciona Mamba na revisão (seção de arquitetura, citando Gu e Dao, 2024) e o artigo de Hu et al. (arXiv 2409.03231, que aplica Mamba a sistemas dinâmicos) apareceu **só em resumo de busca** e não foi aberto: não sei se traz algum conjunto real; pelo resumo, o Duffing dele parece simulado (**não verificado**).
- **Outras SSMs com número no Silverbox:** S4, S5, S5R e LRU em [B] (seção 4.2 acima) e o DS3M [D]. Nenhum é seletivo (Mamba); S4/S5/LRU são lineares invariantes por camada, e o DS3M é um modelo de estado não linear.
- Implicação: o resultado do grupo com Mamba-3 no Silverbox seria, pelo que achei, inédito na literatura consultada; mas a ausência vem de busca limitada e deve ser dita assim no texto ("não encontramos").

### 4.4 Faixas (apenas o que foi verificado)

- **Tradicionais fortes:** multisseno 0,29 a 0,64 mV; arrow full 0,33 a 2,25 mV; sem extrapolação 0,26 a 0,57 mV (NL-LFR, GP-NARX, pNARX); o PNLSS tem 0,26 só no arrow full, em divisão pré-oficial. Os lineares ficam em 6,6 a 14,4 mV.
- **Neurais:** SUBNET e DS3M no melhor caso (0,23 a 0,36 / 0,74 a 1,4 / 0,23 a 0,32 mV); GRU/LSTM/RNN em 1,0 a 1,95 / 2,4 a 6,1 / 0,95 a 1,7 mV; SSMs profundas S4/S5/LRU em 3,6 a 4,7 mV no trecho completo.

---

## 5. Armadilhas

1. **Extrapolação de amplitude.** A amplitude do arrow cresce linearmente até passar a do treino; o treino oficial é só multisseno de amplitude constante. SUBNET atribui à extrapolação quase todo o erro maior no teste completo: sem a região de extrapolação o erro cai de 1,4 para 0,32 mV ([S], seção 3.2, Figura 5). Redes com ativação saturante e normalização pelo treino tendem a sofrer aí; já os modelos com a estrutura cúbica certa (PNLSS, NL-LFR, polinomiais) extrapolam bem. Por isso os três números existem: **não reportar só um**.
2. **"Sem extrapolação" não é igual em todos os artigos.** O carregador usa as primeiras 32.000 amostras do arrow (que começa no índice 100); Bonassi et al. e Tiels usam as primeiras 25.000 de 40.500 ([B], seção 6); Bemporad testa as primeiras 40.000; SUBNET fala em 40.000. Números de [B], [N] e [S] **não são diretamente comparáveis** aos de [C], [P], [M], [D].
3. **Versões diferentes dos mesmos baselines.** pNARX em [C] v2 é 0,640/2,25/0,571, enquanto a planilha (out/2025, código `d9c4972`) dá 0,505/1,41/0,455; GRU 1,49/2,80/0,947 em [C] contra 1,07/4,37/1,17 na planilha. Citar sempre a versão e o código.
4. **Resultados "legados".** Os valores de PNLSS (0,26), redes sigmoidais (0,30), LS-SVM (0,32), Poly-LFR (0,35), GP (0,36) e NL-LFR (0,38) estão marcados na planilha como "publicados antes da divisão oficial". Não sei como cada autor separou treino e teste; tratar como ordem de grandeza.
5. **Janela de inicialização.** Oficialmente o modelo só vê 50 amostras de `y` no começo de cada teste (RMSE medido de n=50 em diante). O protocolo do grupo usa 256. Para a linha "comparável à literatura", usar 50; se o modelo exigir 256, dizer explicitamente que o intervalo avaliado difere (as 206 amostras a mais são ~1% do teste multisseno, derivado), ou relatar também a versão com 256.
6. **Transientes e junções.** Os trechos de multisseno são separados por uma folga entre realizações (o artigo diz 100 zeros; os índices do carregador indicam 500 amostras de folga; **não verificado nos dados**). Janelas de treino que cruzam uma junção, ou janelas de 50 passos que a incluem, contêm um transiente de reinício. Bemporad removeu esses trechos do treino ([N]). Conferir também o início do arrow (o carregador descarta as 100 primeiras amostras).
7. **A realização 8 de multisseno é partida entre treino e teste** (derivado). Não é vazamento de realização completa, mas vale dizer.
8. **Unidades.** Volts e mV misturados (SS-KAN usa V; o resto mV); NRMSE e FIT também circulam. Converter e rotular.
9. **Reprodutibilidade.** Planilha: "resultados podem variar por máquina/execução por causa de não determinismo da GPU" (nota do autor da linha NL-LFR em [P]).
10. **O Silverbox é de baixa ordem e quase sem ruído.** A própria nota diz que a relação sinal-ruído é alta; o "ruído" de H3 não é testado aqui. Em [C] os autores observam que modelos lineares já obtêm ajustes qualitativamente muito bons e que o pNARX fica perto do estado da arte com custo mínimo: o Silverbox favorece os polinomiais.

---

## 6. Como carregar em Python (numpy/scipy apenas)

**Download necessário (peça autorização ao Vinicius antes):**
- URL de visualização: https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view?usp=sharing
- URL direta provável: `https://drive.google.com/uc?export=download&id=17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy` (o mesmo padrão funcionou para a nota técnica de 375 KB; **para o zip não foi testado**; arquivos pequenos costumam não passar pela tela de confirmação do Drive)
- Arquivo: `SilverboxFiles.zip`, ~5,8 MB (5.793.999 bytes segundo o carregador, não confirmado).
- Alternativa oficial: `pip install nonlinear-benchmarks`. Evitar: o grupo não quer dependências novas, e o pacote exige `tqdm`, `progressbar`, `requests`, `rarfile`. A função do pacote só faz o download, o `loadmat` e o fatiamento mostrado abaixo, então dá para reproduzi-la com numpy/scipy.
- Registrar o SHA-256 do zip em `execucao.json` (o plano proíbe versionar o arquivo, só o script e o hash). O hash esperado **não é conhecido**; o primeiro download o define.
- Extrair o zip numa pasta nova e vazia, sem executar nada do conteúdo; rodar o Python com `-I`.

Esboço (derivado do carregador; **não executado**, porque não há arquivo):

```python
import hashlib, urllib.request, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat

URL = "https://drive.google.com/uc?export=download&id=17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy"
FS = 610.35                       # Hz, sampling_time = 1/610.35
N_INIT = 50                       # janela oficial de inicialização

def baixar(pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    zip_path = pasta / "SilverboxFiles.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(URL, zip_path)
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(zip_path.stat().st_size, sha)      # esperado ~5793999; gravar o hash
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pasta)                   # conferir nomes antes se quiser
    return pasta / "SilverboxFiles" / "SNLS80mV.mat"

def carregar(mat_path: Path):
    d = loadmat(mat_path)
    u = np.asarray(d["V1"]).ravel()           # o carregador usa out['V1'][0]
    y = np.asarray(d["V2"]).ravel()
    return u, y

def divisao_oficial(u, y):
    s = int(len(u[40650:127400]) * 0.75)      # 65062
    corte = 40650 + s                         # 105712
    sl = lambda a, b: (u[a:b], y[a:b])
    return {
        "train_val": sl(40650, corte),
        "test_multisine": sl(corte, 127400),
        "test_arrow_full": sl(100, 40575),
        "test_arrow_no_extrap": sl(100, 32100),
    }

def rmse_mv(y_true, y_model, n=N_INIT):
    e = np.asarray(y_true)[n:] - np.asarray(y_model)[n:]
    return 1000.0 * float(np.sqrt(np.mean(e**2)))
```

Verificações a fazer assim que o arquivo existir (nenhuma foi feita): formato e dimensões de `V1`/`V2` (o carregador indexa `[0]`, o que sugere forma `(1, N)`); N total (≥127.400); os comprimentos 65.062 / 21.688 / 40.475 / 32.000; folgas entre as 10 realizações nos índices dados acima; e o teste de sanidade do item 7.

---

## 7. Recomendação para o protocolo do grupo

**Princípio:** rodar o Silverbox em duas camadas sobre o mesmo ajuste de modelo, sem mexer no teste para escolher nada.

**A. Camada "literatura" (simulação livre oficial):**
1. Usar exatamente a divisão da seção 3.1. Subdividir `train_val` em treino e validação **só dentro de `[40650:105712]`**, em blocos contíguos (por ex. os últimos 20% como validação; MGU e SUBNET usam algo equivalente), sem tocar nos três testes. Normalizar com média e desvio do treino.
2. Para cada um dos três testes, inicializar com `y[:50]` e `u` inteiro, simular até o fim e calcular `rmse_mv(y[50:], ŷ[50:])`. Reportar o vetor de três números, em mV, na ordem oficial. Rodar todos os métodos do grupo (ARX, NARX polinomial, FROLS, GRU, Mamba-3) por essa mesma rotina.
3. Se algum modelo exigir 256 amostras de aquecimento, usar as primeiras 256 do teste e declarar que o RMSE é de 256 em diante; ou, preferivelmente, rodar com 50 e 256 e mostrar que a diferença é pequena.
4. **Teste de sanidade antes dos modelos neurais:** o ARX linear do grupo deve dar perto de 6,95 / 14,2 a 14,3 / 6,58 mV ([C] Tabela 3, [P]) e o pNARX algo da ordem de 0,5 a 0,64 / 1,4 a 2,25 / 0,46 a 0,57 mV. Se o ARX desviar muito, o problema é o carregador ou a métrica.
5. A tabela final compara com as faixas da seção 4 e **cita a versão e a fonte de cada linha** (planilha oficial com data, ou artigo). Citar PNLSS, NL-LFR, SUBNET, DS3M, GRU/LSTM e S4/S5 como contexto, com a ressalva de divisão (armadilhas 2 a 4).

**B. Camada "protocolo do grupo" (50 passos a partir de várias origens, 256 medições de aquecimento):**
1. Dentro de cada um dos três testes, escolher origens espaçadas (por ex. passo de 50 ou 100 amostras) começando em 256 depois do início do trecho. Em cada origem: aquecer com as 256 medições anteriores e simular 50 passos. Evitar origens cujas janelas cruzem a junção entre realizações de multisseno e o trecho inicial do arrow.
2. Calcular o erro por origem e agregar com a mesma estatística dos outros cenários do TCC; adicionar NRMSE (RMSE / desvio padrão de `y` no treino) para poder comparar com Duffing simulado e massa-mola.
3. **Separar por amplitude:** classificar cada janela como "dentro da faixa do treino" ou "fora" (por exemplo pelo máximo de |y| na janela contra o máximo de |y| no treino) e reportar os dois grupos; o arrow no extrap. é o equivalente oficial do grupo "dentro". É nessa separação que se verá se a SSM extrapola melhor ou pior que a GRU e o polinomial.
4. As 50 amostras valem ~82 ms (derivado de fs = 610,35 Hz); lembrar que no Silverbox o horizonte curto do grupo é bem mais fácil do que a simulação livre de 21.688 a 40.475 passos. Os dois números não devem ser misturados na mesma tabela.

**C. Texto e pré-registro:**
- Declarar: "Mamba-3 e SSMs seletivas não tinham resultado publicado no Silverbox nas fontes que consultamos", sem afirmar que não existe.
- Pré-registrar que a seleção de hiperparâmetros usa só a validação interna do `train_val`; que os três RMSE oficiais são calculados uma vez por configuração final; e que o `.mat` fica fora do Git.
- Registrar no `execucao.json`: hash do zip, índices dos fatiamentos, `N_INIT`, versão desta nota.

---

## Resumo em cinco linhas

1. **Divisão padrão:** `SNLS80mV.mat`; treino+validação = multisseno `[40650:105712]` (65.062 amostras, a validação você separa dele); testes = multisseno `[105712:127400]`, arrow full `[100:40575]` e arrow sem extrapolação `[100:32100]`; init de 50 amostras.
2. **Métrica:** RMSE em mV de simulação livre, três números `[multisseno; arrow full; arrow sem extrap.]` (não é previsão de um passo).
3. **Tradicionais verificados:** ~0,26 a 0,64 mV no multisseno e no arrow sem extrapolação (NL-LFR 0,29/0,33/0,26; GP-NARX 0,30/0,60/0,26; pNARX 0,5 a 0,64 / 1,4 a 2,25 / 0,46 a 0,57); PNLSS 0,26 só no arrow full, pré-oficial; lineares em 6,6 a 14,4 mV.
4. **Neurais verificados:** SUBNET 0,36/1,4/0,32 e DS3M 0,23/0,74/0,23; GRU/LSTM/RNN em 1,0 a 1,95 / 2,4 a 6,1 / 0,95 a 1,7 mV; S4/S5/LRU 3,6 a 4,7 mV no trecho completo. Nenhum resultado de Mamba encontrado.
5. **Sem verificação:** tamanho real do zip (5.793.999 bytes vem do código do carregador), estrutura e N total dos `.mat`, folga real entre realizações (100 contra 500 amostras), licença dos dados, qualquer resultado NARMAX ou Hammerstein-Wiener, artigo original do PNLSS, e o artigo de Hu et al. sobre Mamba (só em resumo de busca).
