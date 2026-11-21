# Comparação com os kernels oficiais do Mamba-3 (E011) - 2026-11-21

**Resultado:** funcionou. Com o mesmo `state_dict` e a mesma entrada, o módulo oficial `mamba_ssm.Mamba3` (SISO, kernels Triton) e o nosso `Mamba3SISO` (`experimentos/mamba3_referencia.py`) diferem de 0,19 % a 0,38 % na saída (erro relativo L2). Essa diferença tem o tamanho do arredondamento bfloat16 que o kernel oficial faz por projeto. Nosso bloco em float32 fica a menos de 5 × 10⁻⁷ de uma versão em float64.

- Oficial: `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa` (o `main` do repositório também está nesse commit em 21/11/2026), `mamba_ssm` 2.3.2.post1, instalado do clone com `--no-build-isolation`.
- Máquina: RTX 4060 de 8 GB (sm_89), driver 610.88, WSL2 com Ubuntu 24.04, Python 3.12.16 (uv), torch 2.14.1+cu130, Triton 3.8.0, gcc 16.2.0 do conda-forge (usado pelo Triton para compilar seu lançador em C).
- Bloco: `d_model=64, d_state=64, expand=2, headdim=64` (duas cabeças), `rope_fraction=0.5`, `chunk_size=64`, sem `is_outproj_norm`. São os padrões dos dois módulos e a configuração dos ensaios.
- Entrada: lote 32, L = 256 e 512. Há dois casos: (a) pesos aleatórios, com a inicialização do próprio módulo oficial (sementes 0, 1 e 2) e entrada normal padrão; (b) pesos treinados de `resultados/mamba3-v1/` (configuração selecionada `lr3e-3`, sementes 1 a 3), com a entrada do bloco obtida de janelas dos **dados de treino** do E001 passadas pelo `encoder` e pela RMSNorm da rede. O teste não foi lido.
- O `state_dict` tem as mesmas 8 chaves, com os mesmos formatos, nos dois módulos. Os dois carregam com `strict=True`.
- O `step()` oficial não foi usado. Só se compara o `forward` da sequência inteira e o `backward`.

Dados completos, por caso, em `execucao.json`. Script: `experimentos/comparar_kernels_oficiais.py`. Registro da instalação: `planejamento/registro-kernels-oficiais-2026-11-21.md`.

## Precisão de cada lado

| Lado | Precisão |
|---|---|
| Oficial | Parâmetros e entrada em float32. `mamba3_siso_combined` converte Q, K, V, Trap, ângulos e Z para bfloat16 qualquer que seja o tipo recebido. As multiplicações (`tl.dot`) usam operandos bfloat16 com acumulação em float32, e a saída do kernel é bfloat16, convertida para float32 antes de `out_proj`. |
| Nosso, float32 | Tudo em float32, com TF32 desligado (`float32_matmul_precision = highest`). É como o grupo treinou. |
| Nosso, float64 | Referência de alta precisão. A função dos autores ainda força ADT, DT e o ângulo acumulado para float32. |
| Nosso, "só entradas bf16" | Float32, mas com Q, K, V, Trap, ângulos e Z arredondados para bfloat16 antes da função de referência. Imita só as conversões de entrada do kernel oficial. |

## Saída do bloco e gradiente em relação à entrada: oficial × nosso float32

Δ = oficial − nosso. "rel. L2" = ‖Δ‖₂ / ‖ref‖₂. "p95 rel." é a métrica `relative_error` dos testes oficiais: o quantil 95 % de |Δ| / |ref|, só nos elementos com |ref| ≥ 10⁻². Os gradientes vêm de (saída · g).sum(), com g normal padrão de semente fixa, igual nos dois lados. As duas últimas colunas comparam o nosso bloco com a versão float64.

| Pesos | Semente | L | máx \|Δ\| | máx \|Δ\| / máx \|ref\| | rel. L2 | p95 rel. | rel. L2 ∂/∂u | nosso fp32 × fp64, rel. L2 | só entradas bf16 × fp64, rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aleatórios | 0 | 256 | 4.1e-02 | 3.7e-03 | 3.7e-03 | 4.0e-02 | 4.3e-03 | 3.3e-07 | 2.6e-03 |
| aleatórios | 0 | 512 | 4.7e-02 | 4.3e-03 | 3.7e-03 | 4.1e-02 | 4.4e-03 | 3.3e-07 | 2.6e-03 |
| aleatórios | 1 | 256 | 2.1e-02 | 4.2e-03 | 3.8e-03 | 3.6e-02 | 4.2e-03 | 3.4e-07 | 2.6e-03 |
| aleatórios | 1 | 512 | 2.2e-02 | 3.9e-03 | 3.8e-03 | 3.7e-02 | 4.2e-03 | 3.5e-07 | 2.6e-03 |
| aleatórios | 2 | 256 | 1.3e-02 | 4.7e-03 | 3.7e-03 | 3.1e-02 | 4.2e-03 | 3.6e-07 | 2.6e-03 |
| aleatórios | 2 | 512 | 1.3e-02 | 4.0e-03 | 3.7e-03 | 3.2e-02 | 4.2e-03 | 3.7e-07 | 2.5e-03 |
| treinados | 1 | 256 | 7.1e-01 | 4.4e-03 | 2.0e-03 | 2.0e-02 | 3.4e-03 | 1.8e-07 | 1.3e-03 |
| treinados | 1 | 512 | 7.5e-01 | 4.6e-03 | 2.0e-03 | 2.0e-02 | 3.7e-03 | 1.8e-07 | 1.3e-03 |
| treinados | 2 | 256 | 1.6e-01 | 4.6e-03 | 1.9e-03 | 1.9e-02 | 3.2e-03 | 1.6e-07 | 1.2e-03 |
| treinados | 2 | 512 | 2.0e-01 | 5.5e-03 | 1.9e-03 | 2.0e-02 | 3.2e-03 | 1.6e-07 | 1.2e-03 |
| treinados | 3 | 256 | 1.2e+00 | 4.5e-03 | 1.9e-03 | 2.0e-02 | 3.3e-03 | 1.8e-07 | 1.4e-03 |
| treinados | 3 | 512 | 1.4e+00 | 5.0e-03 | 1.9e-03 | 2.0e-02 | 3.6e-03 | 1.8e-07 | 1.3e-03 |

Nos pesos treinados, o máximo absoluto é grande porque a saída do bloco chega a 160–270 em módulo; relativo ao máximo, fica em 0,4–0,55 %.

## Gradientes dos parâmetros e efeito na rede treinada

As três últimas colunas valem só para os pesos treinados. Elas comparam a previsão ŷ (um passo à frente, em metros) da rede inteira (`encoder` → bloco → RMSNorm → `head`) com o nosso bloco e com o bloco oficial, nas mesmas janelas de treino.

| Pesos | Semente | L | maior rel. L2 entre gradientes de parâmetros (qual) | rede: RMS de Δŷ [m] | rede: máx \|Δŷ\| [m] | rede, nosso fp32 × fp64: máx \|Δŷ\| [m] |
|---|---:|---:|---|---:|---:|---:|
| aleatórios | 0 | 256 | 7.2e-03 (`C_norm.weight`) | – | – | – |
| aleatórios | 0 | 512 | 8.6e-03 (`B_bias`) | – | – | – |
| aleatórios | 1 | 256 | 2.1e-02 (`dt_bias`) | – | – | – |
| aleatórios | 1 | 512 | 9.6e-03 (`B_bias`) | – | – | – |
| aleatórios | 2 | 256 | 1.5e-02 (`D`) | – | – | – |
| aleatórios | 2 | 512 | 1.1e-02 (`B_bias`) | – | – | – |
| treinados | 1 | 256 | 9.7e-03 (`in_proj.weight`) | 6.5e-04 | 3.1e-03 | 2.8e-07 |
| treinados | 1 | 512 | 2.2e-02 (`in_proj.weight`) | 6.3e-04 | 2.7e-03 | 3.5e-07 |
| treinados | 2 | 256 | 4.9e-03 (`B_bias`) | 4.0e-04 | 2.0e-03 | 2.3e-07 |
| treinados | 2 | 512 | 3.4e-03 (`in_proj.weight`) | 4.1e-04 | 2.0e-03 | 2.8e-07 |
| treinados | 3 | 256 | 4.4e-02 (`dt_bias`) | 5.2e-04 | 2.4e-03 | 3.3e-07 |
| treinados | 3 | 512 | 1.4e-02 (`in_proj.weight`) | 5.2e-04 | 3.0e-03 | 3.2e-07 |

Nos gradientes dos parâmetros, a diferença entre nosso float32 e o float64 não passa de 2,3 × 10⁻⁶. Duas chamadas seguidas do módulo oficial deram saídas idênticas bit a bit: nesta GPU o kernel foi determinístico.

## Interpretação

1. **A diferença está no kernel oficial, não no nosso bloco.** Nosso float32 fica a ≤ 3,7 × 10⁻⁷ (rel. L2) da versão float64. O oficial fica à mesma distância (1,9–3,8 × 10⁻³) do nosso float32 e do float64: as duas colunas coincidem até o terceiro algarismo em `execucao.json`. Se houvesse um termo errado na nossa reimplementação, o oficial estaria mais perto do float64 do que o nosso. Não está.
2. **O tamanho bate com o bfloat16.** O bfloat16 tem 8 bits de precisão (7 explícitos), com erro relativo de arredondamento de até 2⁻⁸ ≈ 0,4 % por valor guardado. Arredondar só as entradas do kernel (coluna "só entradas bf16") já produz 1,2–2,6 × 10⁻³, cerca de dois terços da diferença total. O resto vem das multiplicações com operandos bfloat16, do ângulo acumulado guardado em bfloat16 e da saída em bfloat16, segundo a decomposição publicada na issue [#1017](https://github.com/state-spaces/mamba/issues/1017) e no PR [#1059](https://github.com/state-spaces/mamba/pull/1059).
3. **Concorda com o relato da issue #1017.** O "3 a 7,5 %" relatado lá usa a métrica p95 dos testes oficiais. Aqui, com essa métrica, o resultado é 1,9–4,1 %. Em rel. L2, um comentário da mesma issue, feito com uma RTX 3080 no mesmo commit, mediu 4,3–7,6 × 10⁻³ com cabeças e entradas diferentes das nossas; aqui deu 1,9–3,8 × 10⁻³. O próprio teste oficial `test_mamba3_siso_combined_batched` usa `rtol = 1e-1` e está com as asserções comentadas no commit fixado.
4. **Gradientes:** em relação à entrada, 0,3–0,4 % (rel. L2), na mesma escala da saída. Nos parâmetros, até 4,4 % (`dt_bias`, semente 3, L=256). Os gradientes ligados a `dt` e aos ângulos são os mais sensíveis no kernel oficial (PR #1059 e seu comentário de 13/10/2026). Não investigamos esse ponto além disso.
5. **Efeito prático:** trocar o nosso bloco pelo oficial na rede treinada muda a previsão de um passo em 4–6,5 × 10⁻⁴ m RMS (máx. 3,1 × 10⁻³ m). O RMSE de um passo dessas redes no teste do E001 é 1,0–1,2 × 10⁻³ m (`resultados/mamba3-v1/resumo.md`), e o ARX chega a 10⁻¹⁵ m. Ou seja, a precisão bfloat16 do kernel oficial equivale a cerca de metade do erro de um passo do modelo. Rodar o modelo do grupo com o kernel oficial não daria os mesmos números: o kernel oficial é a mesma conta, só que com precisão menor.

## O que isto permite afirmar

- A reimplementação calcula a mesma função que o módulo oficial `Mamba3` SISO no commit fixado: mesmos parâmetros, mesma ordem da projeção e mesmo pré-processamento. A diferença medida é do tamanho da aritmética bfloat16 do kernel e se explica por ela.
- **Não** permite afirmar igualdade bit a bit, nem que os números dos ensaios se repetiriam com o kernel oficial, que é menos preciso. Também não cobre o `step()` oficial, nem treino com o kernel oficial, nem tempos (a coerência entre `step_ref` e `fwd_ref` no nosso bloco foi verificada em `resultados/mamba3-verificacao-v1/`).
- Limites desta medida: uma GPU (RTX 4060), lote 32, L de 256 e 512, uma configuração de bloco, sem `Input_States`; a referência float64 mantém ADT, DT e ângulo em float32; o commit oficial não tem as correções de precisão ainda abertas (#1059, #1070).

## Frase sugerida para o relatório

> A implementação foi comparada com os kernels oficiais (`mamba_ssm`, commit `e9594ce`, Triton, RTX 4060), com erro de 0,2 % a 0,4 % (norma L2 relativa) na saída do bloco e de 0,3 % a 0,4 % no gradiente em relação à entrada, para os mesmos pesos e entradas. Essa diferença tem o tamanho da aritmética bfloat16 adotada pelo kernel oficial: em float32, a nossa implementação difere de uma referência em float64 por menos de 5 × 10⁻⁷.
