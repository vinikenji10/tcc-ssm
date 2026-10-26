# TCC — identificação de sistemas com SSMs e Mamba

Repositório do trabalho **Identificação de Sistemas e Controle Preditivo Baseado em Modelos de Espaço de Estados Estruturados (SSMs)**.

**Estado em 26/10/2026:** com o orientador, a tese passou a ser a identificação por SSMs comparada a métodos tradicionais; o MPC deixou de ser objetivo. Além do E001 (planta linear, ARX) e do E002 (Mamba-3 na mesma planta), foram executados o E003 (oscilador de Duffing, com NARX polinomial) e o E004 (Duffing com ruído de medição). Em 50 passos, a rede errou cerca de 1,5 cm nos dois casos (NRMSE de 5,4–5,5%). O NARX polinomial foi quase exato sem ruído; com ruído, ficou equivalente à rede quando recebeu atrasos suficientes, mas mostrou-se numericamente frágil ([diagnóstico](resultados/diagnostico-e004-v1/diagnostico.json)). O relatório 2 está em [relatorio-2/](relatorio-2/).

Repositório público em [vinikenji10/tcc-ssm](https://github.com/vinikenji10/tcc-ssm). Os experimentos são locais e não exigem API, chave ou serviço pago. O Mamba-3 não depende do HuggingFace nem de GPU NVIDIA: veja [Mamba-3 em PyTorch puro](experimentos/README.md#mamba-3-em-pytorch-puro).

## Por onde começar

1. Abra [a página inicial do vault](vault/Inicio.md) para entender o objetivo, o estágio atual e as tarefas do grupo.
2. Leia [o primeiro resultado experimental](resultados/linear-v1/resumo.md).
3. Consulte [as instruções de execução](experimentos/README.md) e [o plano do relatório 2](planejamento/plano-relatorio-2.md).

## Organização

| Pasta/arquivo | Conteúdo |
|---|---|
| `vault/` | Vault do Obsidian: projeto, conceitos, experimentos, tarefas, relatórios e referências |
| `experimentos/` | Código, configuração, dependências e verificações da base experimental |
| `resultados/linear-v1/` | Dados, métricas, modelo ARX, figuras e registro da primeira execução |
| `planejamento/` | Plano detalhado e registro de atividades para o relatório 2 |
| `relatorio-2/` | Relatório 2 em LaTeX (para o Overleaf), figuras e PDF compilado |
| `TCC_SSM_relatorio_1.pdf` | Relatório entregue em 14/09/2026 |

## Abrir no Obsidian

Clone ou baixe o repositório. No Obsidian, escolha **Abrir pasta como cofre/vault** e selecione a pasta **`vault`**, dentro do repositório. Comece pela nota `Inicio`.

O vault usa Markdown, links internos e recursos nativos do Obsidian. O PDF anterior e as figuras principais estão em `vault/Anexos` para que possam ser consultados dentro do aplicativo. Os arquivos completos de dados e código ficam nas pastas técnicas do repositório.

## Reproduzir o experimento

Em Linux, macOS ou WSL2, com Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experimentos/requirements.txt
.venv/bin/python -m pip install -r experimentos/requirements-mamba3.txt
.venv/bin/python -m unittest discover -s experimentos -p 'test_*.py' -v
.venv/bin/python experimentos/experimento_linear.py --output resultados/linear-v2
```

Use um nome de pasta novo para cada execução. A saída existente é preservada. O ambiente `.venv` e os caches são locais; as dependências estão registradas no repositório.

## O que o primeiro resultado significa

O ARX(2,2) reproduziu a planta linear simulada sem ruído com erro próximo da precisão numérica. A avaliação inclui simulação de até 50 passos sem fornecer saídas futuras reais ao modelo. Esse resultado verifica a referência no cenário ideal e prepara a comparação com redes; não é resultado de Mamba-3 ou de controle em malha fechada.

## Próximos marcos

- Validar o ambiente do Mamba-3 e os gradientes de uma rede numérica pequena.
- Comparar previsões no mesmo protocolo de treino, validação e teste.
- Preparar a entrega de **26/10/2026, até 20h**, e a apresentação remota de **27/10**.
- Investigar integração com MPC em novembro, após validar a previsão de vários passos.

As datas vêm das orientações fornecidas à equipe; o horário individual da apresentação deve ser consultado no Moodle. Veja [o cronograma do grupo](vault/01%20Projeto/Cronograma.md).

## Colaboração

Use [o guia do grupo](vault/04%20Organizacao/Como%20colaborar.md). Registre alterações de método, novas execuções e decisões no vault, mantendo resultados observados e propostas separados. Os responsáveis pelas frentes de trabalho ainda devem ser combinados pelo grupo.
