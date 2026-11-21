# Registro do ambiente no PC e portão G0 - 21/11/2026

**Natureza:** registro de execução da Onda 0 (`plano-a3-2026-11-21.md`, seção 5), feito com apoio do assistente. Separa o que foi observado do que é proposta. Nada foi commitado.

## Veredito de G0

| Item | Veredito | Evidência |
|---|---|---|
| (a) Testes passam | **Sim** | 25/25 na CPU antes das novas frentes; 50/50 na CPU no fim do dia (inclui `test_protocolo_v2.py`, de outra sessão); 7/7 testes do bloco em `cuda` |
| (b) E001 reproduz | **Sim, dentro da precisão numérica; não bit a bit** | Mesma escolha, ARX(2,2); entradas idênticas bit a bit; saídas diferem até 1,3e-15 m; métricas da persistência iguais até 1 ulp |
| (c) A rede treina em `cuda` com erro de validação na faixa do E002 | **Sim** | `lr3e-3`, semente 1: 1,2996e-02 m na época 280 (Mac: 1,3008e-02 m na época 270); faixa do E002: 1,30e-02 a 4,25e-02 m |

Ressalvas que o orquestrador precisa decidir antes da Onda 2 (detalhes na seção "Problemas encontrados"):

1. Rodar todos os scripts com `PYTHONUTF8=1`. Sem isso, o Windows grava e lê texto em cp1252.
2. `check_against_reference` do `experimento_mamba3.py` falha no PC. O ARX fica no piso numérico e a tolerância relativa é de 1e-9, então qualquer execução completa (com teste) que use os dados gravados no Mac para de rodar.
3. Com `core.autocrlf=true` o checkout tem CRLF, e os hashes das fontes em `execucao.json` deixam de coincidir com os do Mac.

## Ambiente

| Item | Valor |
|---|---|
| Sistema | Windows 11 Pro 10.0.26300 (`Windows-11-10.0.26300-SP0`) |
| CPU / RAM | Ryzen 5 7600X (6 núcleos, 12 threads) / 32 GB |
| GPU | NVIDIA GeForce RTX 4060, 8 GB (8,0 GiB visíveis ao PyTorch), capacidade 8.9, driver 610.88 (CUDA UMD 13.3), modo WDDM |
| Python | 3.13.5 (MSC v.1943, 64 bits), em `.venv` |
| PyTorch | `torch==2.14.1+cu132` (CUDA 13.2 na roda, cuDNN 9.24.0) |
| Demais | numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2 (inalterados), einops 0.8.2, tzdata 2026.6 |

Lista completa em `experimentos/requirements-pc.txt` (saída de `pip freeze`).

Diferenças em relação ao Mac (Python 3.12.14, `torch==2.14.1`, MPS):

- Mesma versão do PyTorch, 2.14.1, na variante `+cu132`. A página de versões anteriores do PyTorch lista, para 2.14.1, os índices `cu126`, `cu130` e `cu132`. Os três têm roda `cp313-win_amd64`; verifiquei isso em `https://download.pytorch.org/whl/<índice>/torch/`. Escolhi `cu132` porque o driver aceita CUDA até 13.3.
- Python 3.13 em vez de 3.12. Todos os pacotes fixados têm roda para 3.13.
- `tzdata` foi acrescentado. O Windows não tem a base de fusos que `zoneinfo` usa, e sem ela todos os scripts param em `ZoneInfo("America/Sao_Paulo")`.

## Comandos

Instalação (Git Bash, a partir de `tcc-ssm/`; restrições = `pip freeze` do `.venv` base):

```bash
.venv/Scripts/python.exe -m pip freeze > restricoes.txt   # numpy, scipy, matplotlib e dependências
sed 's/^torch==2.14.1$/torch==2.14.1+cu132/' experimentos/requirements-mamba3.txt > req-mamba3-cu132.txt
.venv/Scripts/python.exe -m pip install -r req-mamba3-cu132.txt -c restricoes.txt \
    --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple
.venv/Scripts/python.exe -m pip install tzdata -c restricoes.txt
```

Verificações e execuções (todas com `PYTHONUTF8=1`, exceto as duas primeiras execuções do E001, explicadas adiante):

```bash
.venv/Scripts/python.exe -m unittest discover -s experimentos -p "test_*.py" -v
TCC_DEVICE=cuda .venv/Scripts/python.exe -m unittest discover -s experimentos -p "test_mamba3*.py" -v
.venv/Scripts/python.exe experimentos/experimento_linear.py --output resultados/linear-pc-v3
.venv/Scripts/python.exe experimentos/verificar_mamba3.py --output resultados/mamba3-verificacao-pc-v2
.venv/Scripts/python.exe experimentos/experimento_mamba3.py --smoke --device cuda --output resultados/mamba3-pc-smoke-v1
```

As medições de tempo usam `medir_treino.py`, um invólucro que chama `experimento_mamba3.run(..., validation_only=True)` sem alterar o cálculo. Ele só cronometra `evaluate_network`, fixa `torch.set_num_threads` e grava `medicao-pc.json` com tempos e pico de memória CUDA. Os scripts, as configs curtas e as amostras de `nvidia-smi` (1 s) e de processos `python` no WSL (20 s) estão em `resultados/mamba3-pc-tempo-ferramentas-v1/`. As listas de processos do desktop não foram copiadas, porque o repositório é público.

## Pastas de resultados criadas

| Pasta | Conteúdo | Situação |
|---|---|---|
| `linear-pc-v1` | E001, primeira tentativa | **Incompleta**: parou em `ZoneInfo` antes de instalar `tzdata`. Mantida, sem sobrescrever |
| `linear-pc-v2` | E001 sem `PYTHONUTF8` | Completa; CSV e MD em cp1252 |
| `linear-pc-v3` | E001 com `PYTHONUTF8=1` | Completa; é a reprodução usada na comparação |
| `mamba3-verificacao-pc-v1` | Verificação do bloco, CPU e `cuda` | Verificações válidas; **tempos contaminados**: a GPU estava em 99 % de uso com o E011 rodando no WSL |
| `mamba3-verificacao-pc-v2` | Repetição com a GPU livre | Tempos válidos |
| `mamba3-pc-smoke-v1` | Smoke em `cuda` | **Parou** em `check_against_reference` (ver problema 2) |
| `mamba3-pc-smoke-v2` | Smoke por `smoke_tolerancia.py` | Diagnóstico: fluxo completo em `cuda`, com conferência registrada e `abs_tol=1e-12` m |
| `mamba3-pc-tempo-300ep-v1` | 300 épocas, `cuda` | E011 no WSL durante cerca de 1 min (épocas 41 a 50) |
| `mamba3-pc-tempo-300ep-v2` | Repetição com a GPU livre | Válida |
| `mamba3-pc-tempo-30ep-p{1,3,5}-s*-v1` | Concorrência, 30 épocas | Válidas |
| `mamba3-pc-tempo-cpu-10ep-v1` | 10 épocas em `--device cpu` | Válida |
| `mamba3-pc-tempo-ferramentas-v1` | Scripts, configs curtas, logs, amostras de GPU | Apoio |

`linear-pc-v2`, `linear-pc-v3`, `mamba3-pc-smoke-v2` e `mamba3-verificacao-pc-v2` não estavam na lista inicial. Foram criadas porque a regra proíbe sobrescrever uma pasta e porque as medidas contaminadas tinham de ser repetidas.

## Edições no código

Só `experimentos/verificar_mamba3.py`, para aceitar `cuda`, sem mudar nenhum cálculo:

- `synchronize`: ganhou `elif device == "cuda": torch.cuda.synchronize()`. Antes, só sincronizava em MPS, e os tempos em `cuda` mediriam só o lançamento dos kernels.
- A lista de dispositivos ganhou `cuda` quando `torch.cuda.is_available()`.

`experimento_mamba3.py` aceita `--device cuda` sem alteração, porque o argumento é livre e só o padrão é `mps`/`cpu`. Os testes não foram alterados. Observação: `test_experimento_mamba3.py` não lê `TCC_DEVICE` e roda sempre na CPU; só `test_mamba3_referencia.py` roda em `cuda`.

## (a) Testes

- CPU, logo após instalar: 25 testes, OK, em 10,6 s (`mamba3-pc-tempo-ferramentas-v1/testes-cpu.log`).
- `TCC_DEVICE=cuda`, `test_mamba3*.py`: 7 testes, OK. Repetido no fim do dia: OK.
- CPU no fim do dia, com e sem `PYTHONUTF8=1`: 50 testes, OK. Os 25 adicionais são de `test_protocolo_v2.py`, que outra sessão criou nesse meio-tempo.

## (b) Reprodução do E001

`linear-pc-v3` (PC) contra `linear-v1` (Mac); script em `mamba3-pc-tempo-ferramentas-v1/comparar_e001.py`.

| Grandeza | Resultado |
|---|---|
| Ordem escolhida | ARX(2,2) nos dois |
| Entradas `u` e ids (`dados.npz`) | Idênticos bit a bit |
| Saídas `y` (`dados.npz`) | Diferença absoluta máxima de 1,3e-15 m (escala 1,4 m; cerca de 88 % dos valores diferem em poucos ulps) |
| Estados físicos (diagnóstico) | Diferença absoluta máxima de 2,7e-15 |
| Normalização | Desvio-padrão idêntico; média de `y` difere em 1 ulp |
| Coeficientes do ARX | Diferença absoluta máxima de 4,3e-15. A relativa máxima, 2,8e-01, vem do intercepto (-1,4e-16 contra -1,9e-16, os dois zero numérico); nos outros quatro coeficientes fica em até 5,0e-13 |
| Métricas de teste, persistência | Diferença relativa ≤ 3e-16 (1 ulp) |
| Métricas de teste, ARX | 1,1e-15 a 2,3e-13 m contra 8,8e-16 a 1,8e-13 m; diferença absoluta máxima de 5,1e-14 m (1,1e-13 em NRMSE), toda no piso numérico |
| Validação dos 9 candidatos | Todos abaixo de 3e-13 m nos dois; desempate pela tolerância de 1e-12 m escolhe ARX(2,2) nos dois |

Leitura: as diferenças vêm de bibliotecas numéricas diferentes (BLAS/libm do Windows e do macOS) na discretização e nos mínimos quadrados. A escolha, a ordem e todas as métricas acima do piso numérico coincidem. Os ensaios neurais leem `resultados/linear-v1` (dados do Mac), então essas diferenças não os afetam.

## Verificação do bloco Mamba-3

`mamba3-verificacao-pc-v2` (GPU livre) contra `mamba3-verificacao-v1` (Mac):

| Medida | PC cpu | PC cuda | Mac cpu | Mac mps |
|---|---:|---:|---:|---:|
| Erro relativo máx., step_ref contra fwd_ref | 3,6e-07 | 3,9e-07 | 3,0e-07 | 2,5e-06 |
| Erro relativo máx., aquecimento 256 + 256 passos | 6,3e-07 | 6,5e-07 | 1,3e-06 | 2,7e-06 |
| Saídas antes de t=299 idênticas | sim | sim | sim | sim |
| Gradientes finitos em todos os parâmetros | sim | sim | sim | sim |
| Passo de treino, batch 32 × 512 [s, mediana] | 0,438 | 0,101 | 0,198 | 0,081 |
| Avaliação E001, 545 origens [s, mediana] | 1,42 | 0,69 | 0,65 | 0,47 |

Em `mamba3-verificacao-pc-v1` (GPU ocupada) as verificações deram os mesmos valores e os tempos foram 0,492/0,106 s e 1,54/0,67 s. A CPU do PC é cerca de 2× mais lenta que a do Mac neste código, e a `cuda` é um pouco mais lenta que o MPS.

## (c) Treino em `cuda` e tempos

Config `experimentos/config-mamba3-pc-tempo.json`: a de `config-mamba3.json` com um candidato, `lr3e-3` (o escolhido em `mamba3-v1`), e a semente 1. Comando: `medir_treino.py` com `--validation-only` equivalente, `cuda`, 6 threads.

| Execução | Treino total [s] | Avaliações (30) [s] | s/época, só treino | s/época, com avaliação | Melhor RMSE val. 50 passos [m] | Época |
|---|---:|---:|---:|---:|---:|---:|
| PC `300ep-v1` (E011 ativo por ~1 min) | 351,5 | 24,6 | 1,090 | 1,172 | 1,2996e-02 | 280 |
| PC `300ep-v2` (GPU livre) | 388,9 | 27,3 | 1,205 | 1,296 | 1,2996e-02 | 280 |
| Mac `mamba3-v1`, mesma taxa e semente | 242,7 | - | - | 0,809 | 1,3008e-02 | 270 |

- **Tempo por treino de 300 épocas no PC: 5,9 a 6,5 min**, contra cerca de 4 min no Mac. Em `v1`, o bloco das épocas 41 a 50 levou 33,6 s contra ~10,7 s dos demais; sem ele, o total seria cerca de 328 s. Em `v2`, sem nada visível na GPU, todos os blocos ficaram uniformemente mais lentos (~13 s). A carga é limitada pela CPU (a GPU fica em 36 % a 45 % de uso médio com um processo), e a causa provável é carga de CPU de outras sessões. Não amostrei a CPU, então isso fica como hipótese. Na prática, conte com uma variação de ±10 % entre execuções.
- **Pico de memória por processo:** `torch.cuda.max_memory_allocated` 630 MiB; reservado 948 MiB; no `nvidia-smi`, cerca de 1,07 GB por processo (contexto CUDA incluído), sobre ~1,26 GB do desktop.
- **Determinismo:** `v1` e `v2` deram perdas de treino e RMSE de validação idênticos bit a bit em todas as épocas, apesar da concorrência na primeira.
- **Comparação com o Mac:** a curva de validação coincide em três algarismos (época 270: 1,300e-02 contra 1,301e-02 m; época 300: 1,301e-02 contra 1,302e-02 m). A diferença no melhor RMSE é de 0,1 %. Não se esperava igualdade bit a bit (MPS contra CUDA).

## Concorrência na GPU

Treinos curtos de 30 épocas (avaliação a cada 10), sementes 1 a N, um processo por semente, `torch.set_num_threads(2)` e `OMP_NUM_THREADS=2` por processo. Antes de cada medida, a GPU ficou 15 s abaixo de 4 % de uso, sem `python` no WSL; durante as medidas, nenhum `python` apareceu no WSL.

| Processos | s/época por processo, só treino | Duração total [s] | Vazão somada [épocas/s] | Ganho | Memória total na GPU (`nvidia-smi`) | Uso médio da GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,06 | 38,0 | 0,875 | 1,00× | 2,33 GB (1,07 GB do processo) | 35 % |
| 3 | 1,88 a 1,92 | 65,7 | 1,471 | 1,68× | 4,46 GB (1,06 GB/processo) | 89 % |
| 5 | 3,17 a 3,20 | 111,9 | 1,447 | 1,65× | 6,58 GB (1,06 GB/processo) | 89 % |
| CPU, 1 processo, 6 threads, 10 épocas | 3,55 | 40,3 | 0,271 | 0,31× | - | - |

- A semente 1 deu o mesmo RMSE (6,317e-02 m em 30 épocas) com 1, 3 e 5 processos. Rodar em paralelo não muda o resultado.
- A GPU satura com 3 processos. Com 5, a vazão não cresce e a memória chega a 6,6 de 8 GB.
- Na CPU, uma época leva 3,3× mais que na `cuda` com um processo.

## Problemas encontrados (para o orquestrador)

1. **Codificação de texto.** Sem `PYTHONUTF8=1`, `write_text` e `open` usam cp1252 no Windows. `linear-pc-v2` saiu com `metricas-teste.csv` e `resumo.md` em ISO-8859, e `check_against_reference` leria o CSV UTF-8 do Mac com a chave `Persistência` corrompida. Proposta: exportar `PYTHONUTF8=1` em toda sessão; ou, como mudança de código, passar `encoding="utf-8"` nas leituras e escritas. Fica a critério do orquestrador; não editei.
2. **Conferência das referências no E002.** O ARX recalculado no PC a partir de `modelo-arx.json` e dos dados do Mac dá 8,72e-16, 2,11e-14 e 1,074e-13 m (H = 1, 10, 50), contra 8,83e-16, 2,14e-14 e 1,096e-13 m gravados. As diferenças absolutas vão de 1e-17 a 2e-15 m, mas `rel_tol=1e-9` com `abs_tol=1e-18` rejeita. A persistência confere exatamente (`mamba3-pc-smoke-v2/diagnostico-conferencia-referencias.json`). O problema é a plataforma, não a `cuda`. Proposta: `abs_tol=1e-12` m, a mesma tolerância do desempate do E001. Outra saída: gerar e avaliar tudo no PC, como fará o protocolo v2.
3. **CRLF.** `core.autocrlf=true` e sem `.gitattributes`: os `.py` saem com CRLF e `source_sha256` muda. Exemplo: `experimento_linear.py` tem `cfacc919…` no PC e `6b86d97…` com LF, o mesmo valor gravado em `mamba3-v1`. Proposta: `.gitattributes` com `*.py text eol=lf`, ou calcular o hash com LF.
4. **GPU compartilhada.** Às 14h08, o E011 no WSL levou a GPU a 99 %. O `nvidia-smi` do Windows não mostra processos do WSL; é preciso conferir `wsl -e ps`.

## Recomendação para as grades (proposta)

- **Rodar 3 sementes em paralelo na GPU, com 2 threads cada.** A vazão é 1,68× a de um processo; 5 processos não ganham nada e se aproximam do limite de memória.
- Orçamento: uma rede de 300 épocas leva cerca de 6 a 6,5 min sozinha. Com 3 em paralelo, o lote leva ~10 a 11 min, ou seja, ~3,5 min por treino. Uma grade de 3 taxas × 3 sementes do E002 cabe em ~35 min. Some ±10 % de variação e a avaliação final no teste.
- Reservar núcleos: os treinos na GPU são limitados pela CPU (3 processos × 2 threads usam metade das 12 threads). Os métodos tradicionais na CPU, ao mesmo tempo, devem ficar com no máximo ~4 a 6 threads, e os treinos podem desacelerar; medir na primeira execução real.
- Não usar `--device cpu` para as redes: 3,3× mais lento.
