# Mamba-3 nos dados do E001

**Execução de verificação do fluxo (smoke): não usar como resultado.**

Implementação: Mamba-3 SISO em PyTorch, a partir das funções de referência de `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`, float32, dispositivo `cuda`.
Rede: Linear(2→64), 1 bloco(s) Mamba-3 com RMSNorm e residual, Linear(64→1); 34949 parâmetros.

## Escolha na validação

Critério: média, entre as sementes, do menor RMSE da trajetória de 50 passos na validação. Selecionada: **lr1e-3**.

| Configuração | Semente | Melhor época | RMSE validação, 50 passos [m] | Tempo de treino [s] | Situação |
|---|---:|---:|---:|---:|---|
| lr1e-3 | 1 | 1 | 4.464e-01 | 3 | concluido |

## Teste

| Modelo | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] | NRMSE da trajetória |
|---|---:|---:|---:|---:|
| Persistência | 1 | 4.195e-02 | 4.195e-02 | 8.838e-02 |
| Persistência | 10 | 4.022e-01 | 2.534e-01 | 5.339e-01 |
| Persistência | 50 | 6.582e-01 | 6.839e-01 | 1.441e+00 |
| ARX(2,2) | 1 | 8.719e-16 | 8.719e-16 | 1.837e-15 |
| ARX(2,2) | 10 | 3.951e-14 | 2.105e-14 | 4.435e-14 |
| ARX(2,2) | 50 | 1.718e-13 | 1.074e-13 | 2.264e-13 |
| Mamba-3 (semente 1) | 1 | 7.807e-02 | 7.807e-02 | 1.645e-01 |
| Mamba-3 (semente 1) | 10 | 3.640e-01 | 2.616e-01 | 5.512e-01 |
| Mamba-3 (semente 1) | 50 | 4.477e-01 | 4.100e-01 | 8.639e-01 |

Mamba-3, média entre sementes:

| Horizonte | Média do RMSE da trajetória [m] | Desvio-padrão [m] | Mínimo [m] | Máximo [m] |
|---:|---:|---:|---:|---:|
| 1 | 7.807e-02 | - | 7.807e-02 | 7.807e-02 |
| 10 | 2.616e-01 | - | 2.616e-01 | 2.616e-01 |
| 50 | 4.100e-01 | - | 4.100e-01 | 4.100e-01 |

## Como ler

As referências clássicas foram recalculadas a partir dos modelos gravados no E001 e conferidas contra as métricas daquela execução. As previsões da rede usam as mesmas origens e os mesmos alvos.
Em cada origem, a rede é aquecida com as 256 medições até a origem, inclusive, e depois realimenta as próprias previsões; saídas futuras reais não entram.
Na planta linear sem ruído, o ARX chega à precisão numérica. A pergunta aqui é se a rede aprende a dinâmica e quanto erra, não se supera o ARX.
Com três sementes, a dispersão é indicativa; não sustenta uma conclusão estatística ampla.
Tempos de treino desta implementação de referência; não representam os kernels dos autores.

## Artefatos

- `config.json`, `selecao-validacao.json`: configuração, curvas e escolha.
- `modelos/`: pesos do melhor checkpoint de cada execução.
- `metricas-teste.csv`, `avaliacao-teste.json`, `resumo-sementes.json`: resultados no teste.
- `figuras/`: curvas de treino, previsão de um passo, simulação de 50 passos e erro por horizonte.
- `execucao.json`: versões, hashes e dispositivo.
