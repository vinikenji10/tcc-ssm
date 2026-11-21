# Registro: comparação com os kernels oficiais do Mamba-3 (E011) - 21/11/2026

**Status:** concluído. A comparação funcionou na RTX 4060 pelo WSL2.
**Resultado:** `resultados/mamba3-kernels-oficiais-v1/resumo.md` (tabelas, interpretação e frase para o relatório) e `execucao.json` (todos os números, versões e hashes).
**Esforço:** cerca de 30 min de relógio (13h50 a 14h20), dentro do teto de 3 h. A GPU ficou ocupada por cerca de 40 s em cada uma das três execuções completas e por alguns segundos no teste de fumaça.

## O que foi feito

1. Antes de instalar, li o `pyproject.toml`, o `setup.py` e o README do repositório oficial no commit fixado, além das issues recentes (#1017, #1059, #1068, #1069, #1070, #986, #1079).
2. Montei um ambiente isolado em `~/tcc-e011/` no WSL2, sem sudo.
3. Escrevi `experimentos/comparar_kernels_oficiais.py`, que carrega o mesmo `state_dict` no módulo oficial e no nosso, passa a mesma entrada e compara a saída e os gradientes.
4. Rodei a comparação com pesos aleatórios e com os pesos treinados de `resultados/mamba3-v1/`.

## O que o repositório oficial pede hoje (lido em 21/11/2026)

- O `main` de `state-spaces/mamba` continua no commit fixado, `e9594ce1c732d97440f0332fdc43170a2294dbfa` (23/07/2026, "Fix silent forward-pass corruption on Blackwell GPUs…"). Versão do pacote: `2.3.2.post1`.
- `pyproject.toml`: `torch` sem versão; `triton>=3.5.0`, `tilelang==0.1.8`, `apache-tvm-ffi<=0.1.12`, `quack-kernels>=0.3.4`, `einops`, `transformers`, `ninja`.
- README: instalar o PyTorch antes e depois `pip install … --no-build-isolation`. Na instalação padrão, nenhuma extensão CUDA é compilada (só a `selective_scan_cuda` do Mamba-1, se pedida com `MAMBA_KEEP_CUDA_BUILD=TRUE`). O Mamba-3 SISO é Python + Triton, então o `nvcc` não é necessário.
- #986: `apache-tvm-ffi` 0.1.12 quebra o `import mamba_ssm` com `tilelang==0.1.8`, e o maior valor que funciona é 0.1.9. Fixei `apache-tvm-ffi==0.1.9`. A nota de outubro, que pedia a mesma coisa, continua válida.
- #1068: o `import` também pode quebrar pelo CuTe DSL (`nvidia-cutlass-dsl`) se houver bibliotecas cu12 e cu13 misturadas. Aqui não aconteceu.
- #1069 relata validação no mesmo commit com torch 2.14.1 e Triton 3.8.0 (H100). Por isso escolhi essa combinação, que também é a versão do torch usada pelo grupo no Mac (2.14.1).
- #1017, #1059 e #1070: o kernel SISO converte as entradas para bfloat16 de propósito, e o erro contra a referência é dessa ordem. Detalhes na interpretação do resumo.

## Ambiente

| Item | Versão |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB, sm_89 |
| Driver | 610.88 (KMD, Windows); no WSL, `nvidia-smi` 610.57.01 com CUDA UMD 13.3 |
| WSL2 | Ubuntu 24.04.3, kernel 6.6.87.2-microsoft-standard-WSL2, glibc 2.39 |
| Python | 3.12.16 (python-build-standalone, via uv 0.13.9) |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| Triton | 3.8.0 (o que vem com o torch) |
| mamba_ssm | 2.3.2.post1, do clone em `~/tcc-e011/mamba` no commit `e9594ce` |
| Dependências do mamba_ssm | tilelang 0.1.8, apache-tvm-ffi 0.1.9, quack-kernels 0.6.5, nvidia-cutlass-dsl 4.8.1, einops 0.8.2, transformers 5.24.0, numpy 2.5.5 |
| Compilador C (para o Triton) | gcc 16.2.0, binutils 2.46.1 e sysroot glibc 2.39, do conda-forge, via micromamba 2.10.0 |

A lista completa de pacotes está em `~/tcc-e011/freeze.txt`. O ambiente ocupa 8,7 GB.

## Comandos

Os scripts estão em `~/tcc-e011/scripts/` no WSL. Resumo:

```bash
# sudo pede senha: sem sudo. Sem gcc, sem pip e sem python3-venv no Ubuntu.
# uv autônomo (binário oficial; SHA-256 conferido contra o .sha256 da release)
curl -sSLf -o uv.tar.gz https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-unknown-linux-gnu.tar.gz
# Python e venv
uv python install 3.12
uv venv --python 3.12 ~/tcc-e011/venv
# PyTorch com CUDA, depois as ferramentas de build exigidas por --no-build-isolation
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install setuptools wheel packaging ninja
# mamba_ssm no commit fixado
git clone https://github.com/state-spaces/mamba.git ~/tcc-e011/mamba
git -C ~/tcc-e011/mamba checkout e9594ce1c732d97440f0332fdc43170a2294dbfa
uv pip install --no-build-isolation ~/tcc-e011/mamba "apache-tvm-ffi==0.1.9"
# Compilador C para o Triton (micromamba estático; SHA-256 conferido)
micromamba create -p ~/tcc-e011/cc --override-channels -c conda-forge gcc
export CC=~/tcc-e011/cc/bin/gcc
# Comparação (da raiz do repositório, no WSL)
source ~/tcc-e011/env.sh
python experimentos/comparar_kernels_oficiais.py --out resultados/mamba3-kernels-oficiais-v1
```

`~/tcc-e011/env.sh` define o `PATH` só com pastas do Linux (o WSL anexa as do Windows), os caches do uv e do Triton dentro de `~/tcc-e011/`, `CC` e o venv.

## Problemas encontrados

1. **Download do PyTorch:** a primeira tentativa falhou por tempo esgotado ao baixar `nvidia-nvshmem-cu13==3.4.5` de `pypi.nvidia.com` ("Request failed after 3 retries in 126.3s … operation timed out"). A segunda passou, com `UV_HTTP_TIMEOUT=600`, `UV_CONCURRENT_DOWNLOADS=4` e `UV_HTTP_RETRIES=5`.
2. **Sem compilador C, o Triton não roda.** A instalação do `mamba_ssm` termina sem erro, mas a primeira chamada ao kernel falha com:
   ```
   File ".../triton/runtime/build.py", line 32, in _find_compiler
   RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
   ```
   O Triton compila em tempo de execução um lançador em C, que precisa de `Python.h` (presente no Python do uv) e de um `cc`. Resolvido com o gcc do conda-forge e `CC` apontando para ele, sem sudo. Alternativa para quem tiver sudo: `sudo apt install build-essential`.
3. Erro meu no script: na primeira execução, eu tirava a última amostra de `u` e de `y`, mas no E001 `y` tem uma amostra a mais (`pairs(u, y[:, :-1])`). Corrigi e rodei de novo; os números dos pesos aleatórios não mudaram.
4. Hash de `mamba3_referencia.py`: o checkout no Windows usa CRLF, então o SHA-256 do arquivo no disco (`d0ecc6…`) difere do registrado em outubro (`f96944…`). Com os fins de linha normalizados para LF, o hash é `f969446987d251abdbd420a4914e9958649ca33420bca9933542ea923d0d2e0a`, igual ao do commit e ao de outubro. O script agora grava os dois.

## Saída da execução final

```
aleatório semente 0 L=256: saída rel_l2 oficial×nosso fp32 = 3.75e-03
aleatório semente 0 L=512: saída rel_l2 oficial×nosso fp32 = 3.75e-03
aleatório semente 1 L=256: saída rel_l2 oficial×nosso fp32 = 3.76e-03
aleatório semente 1 L=512: saída rel_l2 oficial×nosso fp32 = 3.76e-03
aleatório semente 2 L=256: saída rel_l2 oficial×nosso fp32 = 3.72e-03
aleatório semente 2 L=512: saída rel_l2 oficial×nosso fp32 = 3.73e-03
treinado semente 1 L=256: bloco rel_l2 = 1.96e-03; rede máx |Δy| = 3.13e-03 m
treinado semente 1 L=512: bloco rel_l2 = 1.97e-03; rede máx |Δy| = 2.67e-03 m
treinado semente 2 L=256: bloco rel_l2 = 1.87e-03; rede máx |Δy| = 1.96e-03 m
treinado semente 2 L=512: bloco rel_l2 = 1.88e-03; rede máx |Δy| = 1.95e-03 m
treinado semente 3 L=256: bloco rel_l2 = 1.90e-03; rede máx |Δy| = 2.42e-03 m
treinado semente 3 L=512: bloco rel_l2 = 1.90e-03; rede máx |Δy| = 2.97e-03 m
Gravado resultados/mamba3-kernels-oficiais-v1/execucao.json em 47.3 s
```

## Pendências e limites

- Não foram verificados o `step()` oficial (não usado, como planejado), o treino com o kernel oficial, comprimentos acima de 512 nem o uso de `Input_States`.
- As correções de precisão ainda abertas no repositório oficial (#1059, #1070) reduziriam a diferença. Não foram aplicadas: a comparação é com o commit fixado, sem alterações.
- O ambiente `~/tcc-e011/` fica no WSL e pode ser apagado (8,7 GB) depois da revisão do grupo. Para repetir, basta `source ~/tcc-e011/env.sh` e o comando acima.
