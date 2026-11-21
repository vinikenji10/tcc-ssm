# Registro: planta do E008, multisseno e configurações do E008 e do E009

**Data:** 21/11/2026. Preparado pelo assistente numa sessão paralela às das Ondas 2, a partir das seções 2, 3.2 e 3.4 de `planejamento/pre-registro-a3.md`. Não muda o pré-registro nem `experimentos/protocolo_v2.py`. Nada foi gravado em `resultados/` e nenhum modelo foi ajustado. Sem commit.

## 1. O que foi feito

| Arquivo | Conteúdo |
|---|---|
| `experimentos/plantas_v2.py` | Planta `saturation_friction` (`m p'' = sat(u) − c p' − k p − Fc·tanh(p'/vε)`, RK4 com subpassos sob segurador de ordem zero, vetorizada sobre as trajetórias) e excitação `multisine`, registradas em `PLANTS` e `EXCITATIONS` ao importar. Comandos `gerar` (o de `protocolo_v2.py`, depois de importar este módulo) e `ruido` (a regra do ruído, só com o treino) |
| `experimentos/test_plantas_v2.py` | 16 testes (seção 6) |
| `experimentos/config-saturacao-atrito-aprbs-v2.json` | E008a: semente 20261122, APRBS do E005, ruído de 0,025 m |
| `experimentos/config-saturacao-atrito-multisseno-v2.json` | E008b: semente 20261123, multisseno, ruído de 0,011 m |
| `experimentos/config-duffing-ruido-0005-v2.json`, `-001-`, `-004-` | E009: cópias de `config-duffing-ruido-v2.json` com desvio de 0,005, 0,01 e 0,04 m; mudam só `scenario`, `description` e `measurement_noise_std` (conferido por teste) |

As configurações do E008 têm os blocos `initial_state`, `data` (exceto o ruído), `evaluation`, `bootstrap` e `grids` idênticos aos de `config-duffing-v2.json` (conferido por teste). Os parâmetros da planta estão no bloco `plant`: `saturation_limit` 1,5 N, `coulomb_friction` 0,3 N, `friction_velocity_scale` 0,01 m/s, `rk4_substeps` 20, mais m, c, k e Ts do E005.

O multisseno: `u[n] = A Σ_{k=1}^{204} cos(2π k n / 2048 + φ_k)`, com `A = RMS·√(2/204)` e `RMS = 2/√3 = 1,1547005383792517 N`. Como os harmônicos são distintos e menores que 1.024, os cossenos são ortogonais no período e o RMS de cada trajetória é exatamente o pedido. O produto `k·n` é reduzido módulo 2.048 em inteiros. f0 = 1/(2.048 × 0,05 s) = 0,009765625 Hz; o harmônico 204 fica em 1,9921875 Hz. As 204 fases são as únicas sorteadas pela excitação (em ordem crescente de k), e a ordem de sorteio por trajetória continua a de `generate_dataset`: estado inicial (posição, depois velocidade), excitação, ruído.

Gerar os dados (quando for a hora; nada foi gerado em `resultados/`):

```bash
PYTHONUTF8=1 .venv\Scripts\python.exe experimentos/plantas_v2.py gerar --config experimentos/config-saturacao-atrito-aprbs-v2.json --output resultados/<pasta-nova>
```

Os três pontos do E009 usam a planta de Duffing e podem ser gerados por `protocolo_v2.py gerar` ou por `plantas_v2.py gerar` (o resultado é o mesmo). Uma geração completa de E008a e de E008b foi feita no diretório temporário da sessão para conferir o caminho de ponta a ponta (45 trajetórias, `load_development` com 20 × 2.048 e 5 × 2.048) e apagada; dela só foram lidos os formatos dos vetores.

## 2. Ruído do E008: a conta

Regra (seção 3.2): desvio = 0,07 × desvio-padrão da posição sem ruído do treino, com dois algarismos significativos, numa geração sem ruído com a mesma semente.

Procedimento (`plantas_v2.noise_rule`): `generate_dataset` com a configuração do cenário e `measurement_noise_std = 0`, em memória; a função devolve só `treino_y_limpo` (e entrada e estados do treino) e descarta o resto sem ler. Validação e teste final dessa geração não foram abertos nem resumidos. O desvio é o populacional (`numpy.std`, ddof = 0) sobre todas as 20 × 2.049 = 40.980 amostras de posição do treino, juntas. Como o ruído é o último sorteio de cada trajetória, as trajetórias sem ruído da geração com ruído são as mesmas (conferido por teste).

| Cenário | Semente | Desvio da posição no treino | × 0,07 | Desvio do ruído gravado |
|---|---|---:|---:|---:|
| E008a (APRBS) | 20261122 | 0,361273 m | 0,0252891 m | **0,025 m** |
| E008b (multisseno) | 20261123 | 0,154198 m | 0,0107939 m | **0,011 m** |

O teste `test_e008_noise_follows_the_rule` refaz a conta e confere o valor gravado em cada configuração.

Conferência do fator: no treino do E005/E006 (semente 20261121, mesma conta) o desvio da posição é 0,270556 m, e 0,02/0,270556 = 0,0739. O "0,07" do pré-registro é esse valor arredondado; foi aplicado como escrito. Com 0,0739 os valores seriam 0,027 m (E008a) e 0,011 m (E008b).

## 3. Subpassos do RK4: convergência

Trajetórias de treino completas (20 × 2.048 amostras, sem ruído) de cada cenário, simuladas com 10, 20, 40 e 80 subpassos e comparadas com 640 subpassos (referência). Erro em posição (m) e velocidade (m/s).

| Cenário | Subpassos | Posição: erro máximo | Posição: RMS do erro | Velocidade: erro máximo |
|---|---:|---:|---:|---:|
| E008a | 10 | 1,9e-5 | 1,6e-6 | 4,3e-5 |
| E008a | **20** | **5,7e-7** | **3,1e-8** | **1,3e-6** |
| E008a | 40 | 9,0e-9 | 1,2e-9 | 2,8e-8 |
| E008a | 80 | 6,6e-10 | 7,9e-11 | 1,8e-9 |
| E008b | 10 | 4,0e-6 | 3,2e-7 | 9,0e-6 |
| E008b | **20** | **8,8e-8** | **7,9e-9** | **2,0e-7** |
| E008b | 40 | 4,9e-9 | 5,6e-10 | 1,1e-8 |
| E008b | 80 | 3,0e-10 | 3,7e-11 | 6,9e-10 |

Razão entre os erros máximos de 20 e de 40 subpassos (contra 80): 68 no E008a e 19 no E008b (a ordem 4 do RK4 daria 16; no E008a a convergência é até mais rápida que a assintótica, sinal de que com 20 subpassos ainda se vê um pouco a região rígida do atrito).

Por que o atrito não é um problema de estabilidade aqui: perto de v = 0 a derivada de `Fc·tanh(v/vε)` é `Fc/vε = 30 N·s/m`, o que dá um autovalor de cerca de −30 s⁻¹; com h = 0,05/20 = 0,0025 s, `h·λ ≈ 0,076`, muito dentro da região de estabilidade do RK4 (cerca de 2,8). A transição de sinal do atrito (largura de cerca de 2vε = 0,02 m/s) é atravessada em poucos subpassos, e é daí que vem o erro residual.

**Veredito: 20 subpassos bastam.** O erro máximo de posição com 20 subpassos é cerca de 44 mil vezes menor que o desvio do ruído no E008a (5,7e-7 m contra 0,025 m) e 125 mil vezes no E008b (8,8e-8 m contra 0,011 m); contra o desvio da posição, cerca de 6e5 e 1,8e6 vezes. Nenhuma emenda é necessária. O teste `test_substeps_converge` repete a comparação 20/40/80 numa geração reduzida e exige erro de 20 subpassos abaixo de um milésimo do ruído.

## 4. Quanto a saturação e o atrito atuam no treino

Só as 20 trajetórias de treino de cada cenário, sem ruído (`plantas_v2.training_activity`).

| Medida (treino) | E008a (APRBS) | E008b (multisseno) |
|---|---:|---:|
| Fração das amostras de entrada com \|u\| > 1,5 N (saturadas) | **23,4 %** | **19,4 %** |
| Fração do tempo com \|v\| < vε = 0,01 m/s | **7,7 %** | **5,4 %** |
| Fração do tempo com \|v\| < 3vε | 12,8 % | 12,8 % |
| RMS da entrada (antes da saturação) | 1,133 N | 1,155 N |
| Maior \|u\| | 2,00 N | 5,30 N |
| Desvio-padrão da posição | 0,361 m | 0,154 m |
| Maior \|p\| | 1,34 m | 0,63 m |
| Maior \|v\| | 2,46 m/s | 1,42 m/s |

Notas:
- Na APRBS (amplitude uniforme em ±2 N), a fração esperada de patamares saturados é 25 %; o observado, 23,4 %, é compatível. No multisseno, o fator de crista leva os picos a 5,3 N e a saturação corta 19,4 % das amostras.
- As duas frações de 12,8 % abaixo de 3vε são uma coincidência de contagem (5.246 amostras em cada cenário); com outros limiares as contagens diferem (abaixo de vε: 3.162 e 2.195; abaixo de 0,05 m/s: 6.733 e 7.774).
- A posição chega a 1,34 m no E008a, bem além do equilíbrio estático da força saturada (1,5/4 = 0,375 m), por ressonância: os patamares de 5 a 40 amostras (0,25 a 2 s) incluem o meio período natural (π/2 ≈ 1,57 s) e o amortecimento é baixo (ζ = 0,1).
- O atrito suavizado não prende a massa: com \|v\| < vε ele age como um amortecimento viscoso de 30 N·s/m, o movimento fica superamortecido e a posição **rasteja** até o equilíbrio, com constante de tempo `(c + Fc/vε)/k ≈ 7,6 s`. Por isso, com amplitude pequena a energia decai mais devagar com atrito do que só com o amortecimento viscoso (o teste de dissipação compara os dois só nos primeiros 5 s, quando \|v\| ≫ vε). É uma propriedade da planta pré-registrada, não do integrador; vale registrar no relatório ao interpretar o E008.

## 5. Pontos do pré-registro que se mostraram ambíguos (nenhum inviável)

1. **Um ruído para o E008 ou um por excitação?** A regra fala em "o treino" e "a mesma semente"; como E008a e E008b têm sementes e treinos diferentes, apliquei a regra a cada cenário (0,025 m e 0,011 m). Se a intenção fosse um único valor (por exemplo, o do E008a), isso exige emenda antes de gerar os dados.
2. **Desvio-padrão "da posição do treino":** usei o desvio populacional de todas as amostras do treino juntas (incluindo a amostra inicial de cada trajetória), não a média dos desvios por trajetória.
3. **O fator 0,07** é o arredondamento de 0,0739 (seção 2). Foi usado como escrito.
4. **RMS do multisseno "igual ao da APRBS":** usei o valor teórico escrito no pré-registro (2/√3 N). O RMS realizado da APRBS no treino do E008a é 1,133 N (o sorteio dos patamares varia em torno do teórico).
5. **Geração pelo comando:** a seção 9, item 1, diz que os dados são gerados por `protocolo_v2.py gerar`. Para o E008 o comando é `plantas_v2.py gerar`, que só importa as plantas novas e chama o mesmo `protocolo_v2.generate_from_config`. Não muda nada nos dados; é só uma diferença de texto.

## 6. Recomendação para quem congelar a seleção do E008 (sem mudança feita aqui)

`protocolo_v2.DEFAULT_CODE_FILES` não inclui `experimentos/plantas_v2.py`, que define a planta do E008. `freeze_selection` já aceita `code_files`; no E008, passar:

```python
code_files=DEFAULT_CODE_FILES + ("experimentos/plantas_v2.py",)
```

Alternativa, se o grupo preferir mudar o padrão (patch em `experimentos/protocolo_v2.py`, linha 61):

```diff
-DEFAULT_CODE_FILES = ("experimentos/protocolo_v2.py", "experimentos/avaliar_teste_final.py",
-                      "experimentos/experimento_linear.py", "experimentos/experimento_duffing.py")
+DEFAULT_CODE_FILES = ("experimentos/protocolo_v2.py", "experimentos/avaliar_teste_final.py",
+                      "experimentos/experimento_linear.py", "experimentos/experimento_duffing.py",
+                      "experimentos/plantas_v2.py")
```

A alternativa muda os hashes de código de toda seleção congelada depois dela, inclusive E005/E006; por isso recomendo a primeira.

## 7. Verificação

```bash
PYTHONUTF8=1 .venv\Scripts\python.exe -m unittest discover -s experimentos -p "test_plantas_v2.py" -v
PYTHONUTF8=1 .venv\Scripts\python.exe -m unittest discover -s experimentos -p "test_protocolo_v2.py" -v
```

Resultado em 21/11/2026 (PC, Python 3.13, NumPy 2.5.4, com `OMP_NUM_THREADS=2`): `test_plantas_v2.py` com 16 testes, OK (cerca de 33 s; a regra do ruído refaz as duas gerações completas em memória); `test_protocolo_v2.py` com 26 testes, OK.

O que os 16 testes cobrem: registro ao importar; com `Fc = 0` e \|u\| < 1,5 N, igualdade com a discretização exata do E001 (`experimento_linear.discretize`) até 1e-10; convergência 20/40/80 subpassos nas duas excitações; equilíbrio em ±1,5/k com entrada constante de 3, −5 e 1,5 N; dissipação de energia com entrada nula (monótona, amostra a amostra; conservação sem atrito nem amortecimento); multisseno com o RMS pedido, espectro só nos harmônicos 1 a 204 com amplitudes iguais, período de 2.048 amostras, reprodutível pela semente e consumindo só as 204 fases; recusa de trajetória com mais ou menos que um período; `generate_dataset` reprodutível com a planta nova e com as mesmas trajetórias sem ruído com e sem ruído; ordem de sorteio estado inicial, excitação, ruído refeita à mão; as cinco configurações passam em `validate_config` e seguem o pré-registro; a regra do ruído reproduz os valores gravados; `plantas_v2.py gerar` num diretório temporário dá a estrutura que `load_development` lê, com o mesmo conteúdo da geração em memória, e recusa pasta existente.
