# Pré-registro dos ensaios da A3 (E005 a E009)

**Situação:** aprovado por Vinicius em 21/11/2026, com as cinco recomendações da tabela abaixo. Ainda não visto pelos outros integrantes nem pelo orientador.
**Data:** 21/11/2026. Preparado com apoio do assistente, a partir da seção 4 de `planejamento/plano-a3-2026-11-21.md`, das hipóteses do relatório 2 (seção "Tese do trabalho") e de `planejamento/levantamento-silverbox-2026-11-21.md`. Nada aqui foi combinado com o grupo ou com o orientador.
**Regra:** nenhum ensaio roda antes de este arquivo ser aprovado e entrar em um commit (portão G1). Depois do commit, os valores abaixo só mudam por uma emenda datada, no fim deste arquivo, escrita antes de qualquer leitura de teste final e com o motivo.
**Código:** `experimentos/protocolo_v2.py` (dados, contrato, métricas, bootstrap, seleção congelada), `experimentos/avaliar_teste_final.py` (leitura única do teste), `experimentos/config-duffing-v2.json` e `experimentos/config-duffing-ruido-v2.json` (espelho legível por máquina destes valores para E005 e E006).

## Decisões que dependem de Vinicius

| # | Decisão | Recomendação | Alternativa |
|---|---|---|---|
| 1 | Teto de termos do NARX polinomial completo e do dicionário do FROLS (seção 5.2) | **2.000 termos** (no máximo 1 parâmetro a cada 20 amostras). Favorece o competidor tradicional, o lado conservador para a tese. Aviso: sabemos pelo diagnóstico do E004 que o NARX(10) de grau 3 (1.771 termos) empatou com a rede; com este teto ele entra na grade | 1.000 termos (1 a cada 40 amostras). Exclui o NARX(10) de grau 3 |
| 2 | Quem é "o melhor método tradicional" em H3 | **O tradicional de menor nota na validação** (ARX, NARX completo, FROLS ou NARX por erro de simulação). Todas as famílias aparecem nas tabelas do teste | Exigir que a rede vença, no teste, cada família tradicional |
| 3 | Correção por vários cenários em H3 ("em pelo menos um cenário") | **Bonferroni sobre os três cenários confirmatórios** (E006, E007 e E008a): a decisão de H3 usa intervalos de 98,33 %. As tabelas descritivas ficam em 95 % | Intervalos de 95 % em cada cenário, sem correção (chance maior de um falso "sustentada") |
| 4 | Sementes das redes (seção 6.4) | **Duas etapas:** cada uma das 6 configurações com as sementes 1 e 2; a escolhida ganha as sementes 3, 4 e 5. São 15 treinos por família e cenário, e o teste tem 5 sementes | Cinco sementes em todas as configurações: 30 treinos por família e cenário (o dobro) |
| 5 | Uso único do teste final (seção 9) | **Uma leitura por cenário**; segunda leitura só por falha técnica antes de existirem métricas, com justificativa no registro de uso e citada no relatório | Proibição absoluta: falha técnica vira cenário perdido |

Decisão de 21/11/2026: valem as cinco recomendações.

## 1. Regras comuns a todos os cenários

- **Divisão por trajetória, antes de qualquer janela:** 20 trajetórias de treino, 5 de validação e 20 de teste final (E005, E006, E008, E009). O Silverbox (E007) tem divisão própria (seção 3.3).
- **Convenção temporal (a do E001):** no instante `t` são conhecidos `y[t]` e `u[t]`; `u[t]` gera `y[t+1]`.
- **Origens:** a partir de `t = 256`, a cada 16 amostras, com espaço para 50 passos. Em trajetórias de 2.048 entradas são 109 origens por trajetória: 545 na validação e 2.180 no teste final.
- **Contrato de modelo, igual para todas as famílias** (docstring de `protocolo_v2.py`): em cada origem o modelo recebe as 256 medições `y[t-255..t]`, as 256 entradas que as produziram, `u[t-256..t-1]`, e as 50 entradas futuras `u[t..t+49]`; devolve `ŷ[t+1..t+50]` por simulação livre. Nenhuma medição posterior à origem chega ao modelo (verificado por teste com `NaN`).
- **Critério de escolha, igual para todas as famílias:** RMSE da trajetória simulada de 50 passos na validação, contra a medição (é o que existe também nos dados reais). Família com várias sementes: média do RMSE entre as sementes.
- **Desempate:** diferenças até `1e-12` m contam como empate numérico; vence o candidato com menos parâmetros e, depois, o primeiro na ordem da grade (regra do E001).
- **Borda da grade:** se a escolha de um tradicional cair na borda superior de uma ordem (16), a grade é estendida uma vez, para 24 e 32, ainda só com a validação. Nas redes, taxa escolhida na borda é registrada como limitação, sem extensão (custo).
- **Divergência:** previsão não finita, ou com |ŷ| maior que 10 vezes a maior |y| medida no treino. Na validação, o candidato é descartado e o descarte fica em `selecao-validacao.json`. No teste final, divergência é resultado: o RMSE fica infinito, a família perde as comparações e nada é removido.
- **Métrica principal dos resultados:** RMSE da trajetória de 50 passos no teste final, contra a medição. Também são gravados: RMSE de trajetória e de ponto final em 1, 10 e 50 passos, NRMSE (denominador: desvio-padrão dos alvos do conjunto) e, nos cenários simulados com ruído, as mesmas métricas contra a saída sem ruído (`summarize_rollouts` do E001, sem mudança).
- **Normalização:** só com o treino (`normalizacao-treino.json`).
- **Resultados antigos não são sobrescritos.** Cada execução grava em pasta nova de `resultados/`.

## 2. Sementes

| Uso | Valor |
|---|---|
| Dados E005, E006 e E009 (mesmas forças e posições sem ruído; o ruído só muda de escala) | 20261121 |
| Dados E008a (patamares) | 20261122 |
| Dados E008b (multisseno) | 20261123 |
| Redes (inicialização e ordem dos lotes) | 1, 2, 3, 4, 5 |
| Bootstrap | 20261129 |

A semente dos dados do E003/E004 (20261026) não é reutilizada. Cada trajetória tem sua semente filha (`SeedSequence.spawn`), registrada em `manifesto.json`.

## 3. Cenários

### 3.1 E005 e E006: Duffing sem e com ruído

| Item | Valor |
|---|---|
| Planta | `m p'' + c p' + k p + k3 p³ = u`, m = 1 kg, c = 0,4 N·s/m, k = 4 N/m, k3 = 20 N/m³; amostragem de 0,05 s; RK4 com 10 subpassos sob segurador de ordem zero (a planta do E003) |
| Estado inicial | posição e velocidade uniformes em ±0,2 m e ±0,2 m/s |
| Excitação | APRBS: patamares de 5 a 40 amostras, amplitude uniforme em ±2 N |
| Dados | 20/5/20 trajetórias de 2.048 entradas (2.049 saídas) |
| Ruído de medição | E005: nenhum. E006: gaussiano branco, desvio de 0,02 m |
| Hipóteses | E005 é o cenário de H2 (classe polinomial certa, sem ruído); E006 é confirmatório de H3 |

### 3.2 E008: não linearidade não polinomial (saturação e atrito)

| Item | Valor |
|---|---|
| Planta | `m p'' = sat(u) − c p' − k p − Fc·tanh(p'/vε)`, com `sat(u) = clip(u, −1,5 N, 1,5 N)`, m = 1 kg, c = 0,4 N·s/m, k = 4 N/m, Fc = 0,3 N, vε = 0,01 m/s; amostragem de 0,05 s; RK4 com 20 subpassos (o teste de convergência dos subpassos é do agente D) |
| Estado inicial | o do E005 |
| E008a (confirmatório de H3) | APRBS do E005 |
| E008b (descritivo) | multisseno de fase aleatória: harmônicos k·f0, f0 = 1/(2.048 × 0,05 s) ≈ 0,00977 Hz, k = 1 a 204 (até cerca de 2 Hz), amplitudes iguais, fases uniformes em [0, 2π) sorteadas por trajetória, RMS igual ao da APRBS (2/√3 ≈ 1,155 N), um período por trajetória |
| Dados | 20/5/20 trajetórias de 2.048 entradas, em cada excitação |
| Ruído | gaussiano branco com desvio de 0,07 vezes o desvio-padrão da posição sem ruído do treino (a relação sinal-ruído do E006), com dois algarismos significativos. O valor é calculado numa geração sem ruído com a mesma semente (as trajetórias sem ruído não mudam) e gravado na configuração antes de qualquer ajuste |

### 3.3 E007: Silverbox (dados reais)

Fonte e índices: `planejamento/levantamento-silverbox-2026-11-21.md`, seção 3.1 (divisão do carregador oficial). **Os índices abaixo são conferidos ao abrir o arquivo, antes de qualquer ajuste; só a estrutura do arquivo pode corrigi-los, e a correção vira emenda.** O download precisa de autorização de Vinicius; o arquivo não entra no Git, só o script e o SHA-256.

| Item | Valor |
|---|---|
| Arquivo | `SNLS80mV.mat`, `V1` = entrada, `V2` = saída, fs ≈ 610,35 Hz |
| Treino e validação | só dentro de `train_val = [40650:105712]`. Segmentos contíguos de 2.048 amostras (2.047 entradas). Validação = os segmentos da realização de multisseno 7; treino = realizações 1 a 6. O pedaço da realização 8 que está em `train_val` não é usado. Segmentos que cruzam a folga entre realizações são descartados |
| Teste final | os três testes oficiais: multisseno `[105712:127400]`, flecha completa `[100:40575]` e flecha sem extrapolação `[100:32100]`, lidos uma vez |
| Camada A (literatura) | simulação livre de cada teste inteiro, iniciada com 50 medições (o contrato com H = 50 e uma só origem). RMSE em mV a partir da amostra 51 (o oficial começa na 50; a diferença de uma amostra é declarada). Os três números na ordem oficial |
| Camada B (protocolo do grupo, H3) | os testes divididos em segmentos de 2.048 amostras; origens e janelas do E005 (256 medições, 50 passos). **Métrica de H3:** RMSE de 50 passos sobre todos os segmentos de teste (multisseno e flecha completa), bootstrap por segmento. Também por grupo: janela dentro ou fora da faixa de |y| do treino |
| Sanidade antes das redes | ARX perto de 6,95 / 14,2 / 6,58 mV na camada A (levantamento, seção 7). Se desviar muito, o erro é do carregador ou da métrica, e nada mais roda até resolver |

### 3.4 E009: varreduras de ruído e de dados (descritivo)

| Varredura | Pontos | Dados |
|---|---|---|
| Ruído | desvio de 0; 0,005; 0,01; 0,02; 0,04 m | Semente 20261121 (trajetórias do E005). Os pontos 0 e 0,02 são o E005 e o E006, sem nova execução. Os três pontos novos têm pastas de dados próprias, com teste final próprio |
| Quantidade de treino | 2, 5, 10, 20 trajetórias | Dados do E006: as primeiras n trajetórias de treino (aninhadas); validação e teste do E006. O ponto 20 é o E006. Os modelos dos pontos 2, 5 e 10 entram como famílias extras na seleção congelada do E006, para que o teste do E006 continue sendo lido uma única vez |

Grade reduzida das varreduras na seção 7.

## 4. Estado inicial dos modelos tradicionais

Cada modelo tradicional é avaliado em duas variantes, e a variante faz parte da grade (escolhida na validação junto com a estrutura):

- **`medido`:** os atrasos de saída na origem são as próprias medições `y[t-ny+1..t]` (o método do E001 ao E004).
- **`janela`:** estimação por mínimos quadrados do erro de simulação na janela de 256 medições.

Método exato da variante `janela`, para um modelo com `ny` atrasos de saída e `nu` atrasos de entrada, na origem `t` com `H = 256`:

1. Seja `m = max(ny, nu − 1)`. As incógnitas são os `ny` valores `z = (ŷ[t−H+1+m−ny], …, ŷ[t−H+m])`.
2. A partir de `z`, o modelo é simulado livremente com as entradas medidas até `ŷ[t]`. Todas as entradas necessárias estão em `u[t−256..t−1]`.
3. Custo: `J(z) = Σ (y[s] − ŷ[s])²` para `s` de `t−H+1+m−ny` a `t`; os próprios valores de `z` entram no custo. Medições anteriores ao primeiro índice (menos de 32) não são usadas.
4. **ARX:** `ŷ` é afim em `z`, e o mínimo é a solução exata de mínimos quadrados (`lstsq`).
5. **NARX (completo, FROLS, erro de simulação):** Levenberg-Marquardt, partindo de `z` igual às medições, com no máximo 20 iterações; para quando a redução relativa de `J` fica abaixo de `1e-9`. Jacobiano analítico ou por diferenças finitas, à escolha do agente A, com teste. Se a simulação na janela deixar de ser finita, a origem volta à variante `medido` e a ocorrência é contada.
6. A previsão de 50 passos parte de `ŷ[t−ny+1..t]` (valores ajustados, não medidos) e das entradas futuras.

Com 256 medições, a rede e os tradicionais recebem exatamente a mesma informação; a escolha entre `medido` e `janela` impede que a estimação piore um tradicional sem que a validação perceba.

## 5. Grades dos modelos tradicionais

### 5.1 ARX

Ordens de saída e de entrada em {1, 2, 4, 8, 16}, todas as combinações (25), com intercepto, mínimos quadrados do erro de um passo (`fit_arx`), cada uma nas duas variantes de estado inicial. Extensão de borda: {24, 32}.

### 5.2 NARX polinomial completo

Todos os monômios de `y[t..t−n+1]` e `u[t..t−n+1]` até o grau `g`, com constante (`fit_narx` do E003): `P = C(2n + g, g)` termos. Grade: `n` em {1, 2, 3, 4, 6, 8, 10, 12, 16}, `g` em {2, 3, 4} (o grau 1 é o ARX), limitada pelo teto `P ≤ min(2.000, ⌊N/20⌋)`, em que `N` é o número de linhas do ajuste. Com o teto de 2.000 ficam 21 candidatos (grau 2: os 9; grau 3: n ≤ 10; grau 4: n ≤ 6), cada um nas duas variantes.

Conta com o número real de amostras. Com 20 trajetórias de 2.048 entradas, `N = 20 × (2.048 − n + 1)`, de 40.960 (n = 1) a 40.660 (n = 16).

| n | g | Termos P | Linhas N | Matriz N×P em float64 | N/P | Entra (teto 2.000) |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 2 | 561 | 40.660 | 174 MiB | 72,5 | sim |
| 8 | 3 | 969 | 40.820 | 302 MiB | 42,1 | sim |
| 10 | 3 | 1.771 | 40.780 | 551 MiB | 23,0 | sim |
| 12 | 3 | 2.925 | 40.740 | 909 MiB | 13,9 | não |
| 4 | 4 | 495 | 40.900 | 154 MiB | 82,6 | sim |
| 5 | 4 | 1.001 | 40.880 | 312 MiB | 40,8 | (fora da grade de n) |
| 6 | 4 | 1.820 | 40.860 | 567 MiB | 22,5 | sim |
| 8 | 4 | 4.845 | 40.820 | 1.509 MiB | 8,4 | não |
| 16 | 4 | 58.905 | 40.660 | 17,8 GiB | 0,7 | não |

Justificativa do teto:
- **Memória.** `fit_narx` monta o produto `z[:, termos]` com `N × P × g` valores antes da matriz de regressão, e `lstsq` copia a matriz. O pico estimado é `N × P × 8 B × (g + 3)`: 3,3 GB no maior candidato de grau 3 e 4,0 GB no de grau 4, em 31 GB de RAM, com folga para o treino das redes ao mesmo tempo. O grau 4 completo com 16 atrasos (17,8 GiB só a matriz) não cabe; é ele que torna o teto necessário.
- **Amostras por parâmetro.** Com ruído branco no erro de equação, o excesso esperado de erro quadrático de um passo devido à estimação é cerca de `σ² P/N`. Com `N/P ≥ 20`, fica abaixo de 5 % da variância do ruído.
- **Condicionamento.** As colunas são escaladas (como no E003) e `lstsq` usa o corte padrão, `rcond = ε·max(N, P) ≈ 9·10⁻¹²`: direções com número de condição acima de cerca de `10¹¹` são truncadas. O número de condição escalado e o posto ficam registrados por candidato; candidato com posto incompleto é avaliado (solução de norma mínima) e marcado.
- O teto não usa os dados nem a planta. A ressalva da decisão 1 vale: o diagnóstico posterior do E004 é conhecido.

Com o teto alternativo de 1.000: grau 3 até n = 8 e grau 4 até n = 4 (19 candidatos).

### 5.3 NARX com FROLS

- **Dicionários:** para cada grau, o maior dicionário completo da grade de 5.2 sob o teto: (n = 16, g = 2, 561 termos), (n = 10, g = 3, 1.771) e (n = 6, g = 4, 1.820). Com o teto de 1.000: (16, 2), (8, 3) e (4, 4).
- **Algoritmo:** FROLS de Billings, com Gram-Schmidt modificado no treino. A cada passo entra o termo de maior taxa de redução de erro (ERR). Termo cuja norma ortogonalizada fique abaixo de `10⁻¹⁰` vezes a original é pulado (dependente).
- **Critério de parada do caminho:** 50 termos, ou soma das ERR com `1 − ΣERR < 10⁻¹²`.
- **Escolha:** o número de termos é escolhido na validação entre {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50} (prefixos do caminho), nos três dicionários e nas duas variantes de estado inicial. Os coeficientes de cada prefixo são de mínimos quadrados nos termos escolhidos.

### 5.4 NARX por erro de simulação

- **Ponto de partida:** o modelo FROLS escolhido na validação (termos e coeficientes do erro de equação). Os termos não mudam.
- **Custo de treino:** soma dos erros quadráticos das simulações de 50 passos a partir de todas as origens do treino (de `t = 256`, a cada 16 amostras), com atrasos iniciais medidos.
- **Otimização:** Levenberg-Marquardt, no máximo 100 iterações, parada quando a redução relativa do custo fica abaixo de `10⁻⁸`. Passo que torne alguma simulação não finita é rejeitado (aumenta o amortecimento). Se o custo final não for menor que o inicial, fica o ponto de partida, registrado.
- **Escolha:** o modelo refinado é avaliado nas duas variantes de estado inicial e compete como família própria ("NARX-SE").
- O NARX completo não é refinado: com até 2.000 parâmetros, o jacobiano de cerca de 109 mil resíduos não cabe no tempo do plano.

**Melhor tradicional** (decisão 2): entre ARX, NARX completo, FROLS e NARX-SE, o de menor nota na validação, com o desempate da seção 1. A persistência é sempre relatada, mas não compete.

## 6. Redes

### 6.1 Entrada, saída e avaliação

As duas redes recebem `(u[t], y[t])` normalizados pelo treino e preveem `y[t+1]`. Na avaliação seguem o contrato: aquecimento com as 256 medições em modo de previsão (pares `(u[s], y[s])`, `s = t−255..t`) e, depois, simulação livre realimentando as próprias previsões.

### 6.2 Arquiteturas

| Rede | Estrutura | Parâmetros |
|---|---|---|
| Mamba-3 | a do E003/E004: Linear(2→64), um bloco Mamba-3 SISO (d_state 64, expand 2, headdim 64) com RMSNorm e conexão residual, Linear(64→1) | 34.949 (E004) |
| GRU | GRU de uma camada, entrada 2, estado de 106, seguida de Linear(106→1) | 3·106·(2+106) + 6·106 + 107 = **35.087** (+0,4 %) |

O agente B confere as duas contagens com `sum(p.numel())`. Se a do Mamba-3 for outra, o estado da GRU passa ao inteiro que deixa a contagem mais próxima.

### 6.3 Treino

| Item | Valor |
|---|---|
| Taxas de aprendizado | {1e-3, 3e-3, 1e-2} |
| Perda de um passo | erro quadrático de um passo em janelas de 512 amostras, passo de 128 entre janelas, estado zerado em cada janela, as primeiras 128 posições fora da perda (a do E002); até 1.000 épocas |
| Perda de vários passos | para cada origem de treino (de `t = 256`, **a cada 64 amostras**: 28 por trajetória, 560 por época com 20 trajetórias): aquecimento com 256 medições e simulação livre de 50 passos com gradiente através da realimentação; erro quadrático médio nos 50 passos simulados, em unidades normalizadas; treino desde o início (sem partir do modelo de um passo); até 300 épocas. O espaçamento de 64 (e não 16) vem da medida de tempo da seção 10 e vale igualmente para a GRU |
| Comuns | AdamW, decaimento de pesos 0, recorte de gradiente em 1,0, taxa com cosseno até o máximo de épocas, lotes de 32 |
| Parada pela validação | a cada 10 épocas, RMSE de 50 passos nas 545 origens da validação; guarda o melhor ponto; para depois de 10 avaliações seguidas sem melhora (100 épocas) |
| Teto de tempo por treino | 60 min (um passo) e 90 min (vários passos), cerca de 3 vezes o tempo medido de um treino sozinho; treino que bate no teto fica com o melhor ponto até ali e é registrado como "limite de tempo" |
| Paralelismo na GPU | até 3 treinos ao mesmo tempo, 2 threads de CPU cada (medida de G0: mesmo resultado bit a bit com 1, 3 e 5 processos) |

A grade por família tem 6 configurações (3 taxas × 2 perdas).

### 6.4 Sementes e escolha (decisão 4)

1. Cada configuração treina com as sementes 1 e 2. Nota da configuração: média, entre as duas, do melhor RMSE de validação.
2. Configuração com alguma semente sem ponto válido (todas as avaliações divergentes) é descartada e registrada.
3. A configuração de menor nota treina também com as sementes 3, 4 e 5. Se alguma delas não tiver ponto válido, a configuração é descartada e a seguinte da etapa 1 ocupa o lugar (registrado).
4. No teste entram as 5 sementes da configuração escolhida, cada uma no seu melhor ponto de validação.

## 7. Grade reduzida das varreduras (E009)

| Família | Em cada ponto da varredura |
|---|---|
| ARX | grade completa (5.1) |
| NARX completo | não roda (o FROLS cobre a família polinomial) |
| FROLS | só o dicionário escolhido no E006; se ele passar do teto `⌊N/20⌋` do ponto, o maior dicionário do mesmo grau sob o teto. Número de termos escolhido na validação, como em 5.3 |
| NARX-SE | a partir do FROLS escolhido no ponto |
| GRU e Mamba-3 | a configuração (taxa e perda) escolhida no E006, sem nova busca, sementes 1, 2 e 3, parada pela validação do ponto |

Em todos os pontos, as duas variantes de estado inicial e a escolha pela validação continuam valendo.

## 8. Incerteza e critérios de decisão

- **Bootstrap por trajetória:** 10.000 reamostragens, semente 20261129. Em cada uma, as 20 trajetórias de teste são sorteadas inteiras, com reposição, as mesmas para todas as famílias (diferença pareada).
- **Redes (sementes e trajetórias):** em cada reamostragem também se sorteiam, com reposição, as 5 sementes, independentemente por família. O RMSE de cada semente é calculado nas trajetórias sorteadas e a nota da família é a média sobre as sementes sorteadas. A estimativa pontual é a média, entre as sementes, do RMSE em todas as trajetórias. O desvio entre sementes também é relatado.
- **Intervalo:** percentis empíricos de 2,5 % e 97,5 % (`numpy.quantile`, `method="inverted_cdf"`), para o RMSE de cada família e para cada diferença pareada.
- **Regra de equivalência (H3):** a diferença rede − competidor decide só se o intervalo inteiro estiver de um lado do zero. Abaixo: rede melhor. Acima: competidor melhor. Contendo o zero: equivalentes.
- **H1:** em cada cenário, NRMSE da trajetória de 50 passos no teste ≤ 10 % para o Mamba-3 (média entre as 5 sementes), sem valores não finitos em nenhuma semente (texto do relatório 2). **Referência do erro:** nos cenários simulados, a saída sem ruído, como nos 5,4 a 5,5 % do relatório 2 (contra a medição, só o ruído do E006 já vale cerca de 7 % e o limiar deixaria de medir o modelo). No Silverbox, a medição, que é a única disponível. H1 vale para E005, E006, E007 e E008a; nos pontos do E009 o NRMSE é só relatado.
- **H2 (E005):** o melhor tradicional tem RMSE menor que o Mamba-3 (intervalo da diferença rede − tradicional acima de zero) e custo de ajuste ao menos 100 vezes menor. Custo: tempo de relógio do procedimento completo de escolha da família (grade e validação), no mesmo PC, redes na GPU e tradicionais na CPU (E010).
- **H3:** sustentada num cenário confirmatório (E006, E007 camada B, E008a) se o Mamba-3 tiver RMSE menor que o melhor tradicional **e** que a GRU, as duas diferenças com intervalo abaixo de zero. Com a decisão 3, esses intervalos são de 98,33 % (`1 − 0,05/3`). H3 é sustentada no trabalho se valer em pelo menos um desses cenários. E005, E007 camada A, E008b e E009 são descritivos.
- **Métrica de H3:** RMSE da trajetória de 50 passos contra a medição. Nos cenários simulados, o mesmo cálculo contra a saída sem ruído é relatado ao lado, mas não decide.

## 9. Teste final: uso único

1. Os dados de cada cenário são gerados uma vez, por `protocolo_v2.py gerar`. O teste final fica em `teste-final/`, separado do desenvolvimento, com o SHA-256 no manifesto.
2. Todo o desenvolvimento (ajuste, escolha, depuração e figuras de desenvolvimento) usa só `dados-desenvolvimento.npz`, por `load_development`.
3. As escolhas de cada cenário são gravadas num arquivo de seleção congelada (`freeze_selection`): escolha e nota de validação por família, adaptador, membros (sementes), hashes dos artefatos, do código, da configuração e dos dados. O arquivo não é sobrescrito; mudança exige outro arquivo.
4. **Nenhum teste final é lido antes de todas as seleções dos cenários que couberam no prazo estarem congeladas e em commit** (portão G4).
5. O teste final só é aberto por `load_final_test`, chamado por `avaliar_teste_final.py`. A função confere a seleção e os hashes e grava uma linha em `teste-final/uso-teste-final.jsonl` (data, chamador, hash da seleção) antes de abrir o arquivo. Os modelos são construídos antes, para que um erro de adaptador não consuma o uso.
6. Uma leitura por cenário. A segunda leitura exige justificativa (decisão 5), que fica no registro e vai para o relatório.
7. Depois da leitura, nenhum modelo é reajustado. Erro de código descoberto depois do teste é corrigido, relatado e mostrado ao lado do resultado original, que não é apagado.
8. A proteção é de procedimento: o arquivo pode ser aberto por fora, mas o registro de uso, os hashes e os commits tornam o uso verificável. Nenhum agente de desenvolvimento lê `teste-final/`.

## 10. Orçamento de tempo

Medidas de 21/11 na RTX 4060, com `torch 2.14.1+cu132`. Um passo, Mamba-3: treino real de 300 épocas nos dados do E001 (`planejamento/registro-ambiente-pc-2026-11-21.md`). Demais: tempo de uma atualização em dados aleatórios, com as redes da seção 6.2, medido pelo orquestrador; são estimativas de ordem de grandeza, não ensaios.

| Item | Medida | Por treino, sozinho na GPU |
|---|---|---|
| Mamba-3, um passo | 1,2 a 1,3 s por época, com avaliação (9 atualizações de 0,11 s) | até 1.000 épocas: cerca de 21 min |
| Mamba-3, vários passos | 0,36 s por atualização de 32 origens (1,05 s com 128; custo proporcional ao lote) | 18 atualizações por época, 6,5 s; até 300 épocas: cerca de 32 min |
| GRU, um passo | 0,003 s por atualização | até 1.000 épocas: 1 a 3 min, dominado pelas avaliações |
| GRU, vários passos | 0,036 s por atualização de 32 origens | até 300 épocas: cerca de 4 min |
| Três treinos de um passo ao mesmo tempo (Mamba-3) | vazão 1,68 vez a de um processo; cinco não ganham mais | não medido para vários passos |

Com as origens de treino a cada 16 amostras, como no primeiro rascunho, a perda de vários passos do Mamba-3 levaria cerca de 2 h por treino (68 atualizações por época) e os cenários não caberiam na janela. Por isso o espaçamento passou a 64 (seção 6.3), antes de qualquer execução.

| Bloco | Treinos | Pior caso, em sequência |
|---|---|---|
| Mamba-3, um cenário completo | 6 de um passo, 6 de vários passos, mais 3 da configuração escolhida | 6 × 21 + 6 × 32 + 3 × 32 min ≈ 7 h |
| GRU, um cenário completo | os mesmos 15 | menos de 1 h |
| Cinco cenários completos (E005, E006, E007, E008a, E008b) | 150 | cerca de 40 h |
| E009 (6 pontos novos, 3 sementes, configuração do E006) | 36 | cerca de 11 h |
| **Total** | **186** | **cerca de 50 h de GPU**, sem contar a parada pela validação nem o paralelismo, que reduzem |
| Janela de GPU | seg 23/11 a sex 27/11 | 120 h de relógio se o PC ficar ligado; folga de pelo menos 2 vezes |

**Tradicionais na CPU: ainda sem medida**, porque o código é da Onda 2. O tempo por família é medido numa execução curta ao fim da Onda 2 e entra aqui por emenda, antes de qualquer ensaio. Se não couber, vale o item 5 das regras de corte.

## 11. Regras de corte

Do plano (seção 3), sem mudança:
- Blocos 1 a 4 (protocolo, tradicionais fortes, GRU e Mamba-3, E005/E006) não fechados até **qua 25/11**: cortar o E009 e a comparação com os kernels oficiais (E011).
- Silverbox não fechado até **sex 27/11**: cortar o E008 e relatar o Silverbox no estado em que estiver.
- **Dom 29/11, fim do dia:** encerrar experimentos, sem exceção. O que não rodou vira limitação declarada.

Se a conta da seção 10 não couber na janela, cortar nesta ordem, antes de começar, e registrar:
1. Máximo de épocas pela metade (1.000 → 500 e 300 → 150), com a mesma parada pela validação.
2. E008b (multisseno) fora; fica só o E008a.
3. Redes fora do E009 (as curvas ficam só com os tradicionais).
4. E009 reduzido a ruído {0,01; 0,04} m e treino {5, 10} trajetórias.
5. Na CPU: se a estimação do estado inicial do NARX completo não couber, a variante `janela` roda só nos 3 melhores candidatos da variante `medido`.

As sementes do teste (5 por rede) e as grades de E005 a E008a não são cortadas por tempo; se nem assim couber, o cenário inteiro sai e vira limitação.

## 12. Afastamentos do plano e decisões de detalhe

- **Sementes em duas etapas** (decisão 4): o plano fala em cinco sementes; o teste continua com cinco, mas a busca usa duas.
- **Estado inicial em duas variantes:** o plano pede estimação na janela de 256; aqui ela é uma das duas variantes, escolhidas na validação, para não piorar um tradicional às cegas.
- **Perda de vários passos treinada desde o início**, com 300 épocas no máximo e origens de treino a cada 64 amostras (o plano não fixava; o espaçamento veio da medida de tempo).
- **Hashes de arquivos de texto com fins de linha em LF** (`sha256_text`), para que a seleção congelada feita no Windows confira no macOS dos outros integrantes.
- **Codificação:** todas as execuções no PC usam `PYTHONUTF8=1`.
- **GRU sem camada de entrada**: a projeção Linear(2→64) antes da GRU seria redundante com os pesos de entrada da própria GRU; os parâmetros ficam na recorrência.
- **NARX por erro de simulação só a partir do FROLS** (seção 5.4, por custo).
- **E009 reaproveita E005 e E006** como pontos da varredura, e os modelos da varredura de dados entram na seleção do E006 para não reler o teste.
- **Extensão única da grade na borda** para os tradicionais (crítica do relatório 2 às grades na borda).
- **Métrica de decisão contra a medição**, não contra a saída sem ruído (a do E004): é a única disponível no Silverbox e a mais conservadora (o piso de ruído é igual para todos).
- **Teto do NARX dependente de N** (`⌊N/20⌋`), para que a varredura de dados com 2 trajetórias não tenha mais termos que amostras.
- No código, os nomes de funções e variáveis seguem o inglês do repositório (`fit_arx`, `summarize_rollouts`); docstrings, mensagens e arquivos estão em português.

## 13. Arquivos e verificação

- `experimentos/protocolo_v2.py`: dados com teste reservado (planta e excitação como funções), contrato de modelo, métricas, divergência, validação, bootstrap, seleção congelada, acesso ao teste e adaptadores registrados (persistência e ARX com atrasos medidos).
- `experimentos/avaliar_teste_final.py`: leitura única do teste final, métricas, intervalos e veredito de H3.
- `experimentos/test_protocolo_v2.py`: divisão sem interseção, reprodutibilidade (e igualdade com o gerador do E003), acesso ao teste só com seleção congelada e uso registrado, bootstrap em casos de resposta exata, ausência de medições futuras (`NaN`), alinhamento temporal, métricas iguais às de `summarize_rollouts` e do E004, divergência, e `avaliar_teste_final.py` de ponta a ponta com persistência e ARX.

```bash
.venv\Scripts\python.exe -m unittest discover -s experimentos -p "test_protocolo_v2.py" -v
```

## Emendas

Nenhuma.
