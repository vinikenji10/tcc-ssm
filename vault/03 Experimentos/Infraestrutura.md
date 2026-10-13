---
tipo: infraestrutura
status: hardware-identificado-execucao-neural-pendente
atualizado: 2026-10-12
tags: [tcc, hardware, mamba]
---

# Infraestrutura

## Máquinas disponíveis

| Máquina | Configuração | Evidência atual |
|---|---|---|
| MacBook Air | Apple M5, 16 GB de memória unificada | Identificado localmente; base clássica executada em CPU |
| PC | RTX 4060, 8 GB VRAM; Ryzen 5 7600X; 32 GB DDR5 | Configuração informada pelo usuário; ainda sem teste remoto |

## Estratégia proposta

Usar o Mac para desenvolvimento, dados, gráficos e documentação. Priorizar o PC para o treinamento da rede pequena. A capacidade de memória parece adequada para o recorte, mas o pico real e a compatibilidade dos kernels precisam ser medidos.

Se o PC usar Windows, avaliar Linux em WSL2 para o caminho CUDA. Não houve instalação ou configuração desse PC nesta fase.

## Suporte do Mamba-3

O caminho oficial usa Linux e infraestrutura NVIDIA/CUDA; não tem suporte direto equivalente para a GPU Apple. Há adaptações comunitárias em MLX para Apple Silicon, que precisam ser verificadas contra a formulação e uma referência confiável.

O módulo oficial distingue processamento de sequência e atualização por amostra. A documentação de `step` informa teste apenas em H100. Esse aviso pertence ao caminho otimizado da implementação, não a uma exigência universal da arquitetura matemática.

Há relato de uso de Mamba-3 SISO em RTX 4060, acompanhado de divergências numéricas. Isso é evidência de uma execução relatada, não certificação da nossa configuração ou da versão atual.

## Fontes consultadas em 12/10/2026

- [Código oficial Mamba-3](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py).
- [Repositório oficial](https://github.com/state-spaces/mamba).
- [Adaptação MLX comunitária](https://github.com/Jada42/mlx-mamba3).
- [Relato de execução em RTX 4060](https://github.com/state-spaces/mamba/issues/1017).
- [CUDA em WSL, NVIDIA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Verificações antes de medir desempenho

Identificar versões e commit; executar a rede; calcular gradientes; comparar resultados com uma referência; confirmar aquecimento e estado; só depois medir latência. Se a implementação for adaptada, registrar essa origem nos resultados.

Veja [[E002 - Mamba-3]] e [[SSMs e Mamba]].

