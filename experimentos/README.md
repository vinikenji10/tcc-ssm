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

## Próximo passo com Mamba-3

1. Identificar a máquina/GPU que será usada e validar importação, passagem direta e gradientes do código oficial.
2. Treinar uma rede de regressão numérica com `treino_u` e `treino_y`, usando a normalização do treino.
3. Selecionar configurações na validação; documentar janelas, aquecimento e reinício de estado.
4. Avaliar nas mesmas trajetórias e origens. Adaptar o preditor para aquecer o estado usando apenas o passado e realimentar saídas previstas no horizonte.
5. Comparar métricas e custos sob regimes explicitamente documentados.

A interface `predict_step` das referências ainda não implementa uma rede com cache de estado. A adaptação para Mamba-3 precisa ser escrita e verificada. A presença desta base não confirma suporte dos kernels ao hardware futuro.
