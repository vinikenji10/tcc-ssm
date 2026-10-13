---
tipo: experimento
id: E001
status: executado-localmente
data_execucao: 2026-10-12
atualizado: 2026-10-12
tags: [tcc, experimento, arx]
---

# E001 — referências lineares

**Pergunta:** a base de simulação, identificação e avaliação consegue reproduzir um sistema linear conhecido a partir de sinais de entrada e saída?

**Resultado:** sim, nesta execução ideal sem ruído. ARX(2,2) alcançou erro próximo da precisão numérica. Não houve execução de Mamba-3 ou MPC.

## Método

Aplicamos o [[Protocolo comum]] à planta massa–mola–amortecedor. A persistência usa a posição atual como previsão; ARX é ajustado exclusivamente aos sinais de treino. A validação selecionou ordem 2 para saídas e entradas.

O teste contém cinco trajetórias independentes e 545 origens por modelo. Durante a simulação, só entram medições até a origem; depois, são realimentadas previsões. Consulte [[Previsao e simulacao]].

## Resultados de teste

| Referência | Horizonte [passos] | RMSE final [m] | RMSE da trajetória [m] |
|---|---:|---:|---:|
| Persistência | 1 | 4,195 × 10⁻² | 4,195 × 10⁻² |
| Persistência | 10 | 4,022 × 10⁻¹ | 2,534 × 10⁻¹ |
| Persistência | 50 | 6,582 × 10⁻¹ | 6,839 × 10⁻¹ |
| ARX(2,2) | 1 | 8,832 × 10⁻¹⁶ | 8,832 × 10⁻¹⁶ |
| ARX(2,2) | 10 | 4,024 × 10⁻¹⁴ | 2,142 × 10⁻¹⁴ |
| ARX(2,2) | 50 | 1,754 × 10⁻¹³ | 1,096 × 10⁻¹³ |

ARX tem cinco parâmetros incluindo o intercepto. O ajuste foi em CPU; o tempo registrado não constitui benchmark de inferência.

## Figuras

### Entrada e saída

![[E001-entrada-saida.png]]

### Previsão de um passo

![[E001-previsao-um-passo.png]]

### Simulação de 50 passos

![[E001-simulacao-50-passos.png]]

### Erro por horizonte

![[E001-erro-por-horizonte.png]]

Os trechos foram predefinidos na primeira trajetória de teste. A curva agregada e a tabela utilizam todas as origens de teste. Erros muito pequenos devem ser interpretados como precisão numérica, sem um significado prático para suas diferenças.

## Verificações

Oito verificações passaram: discretização contra RK4 independente; equilíbrio físico; identificação e simulação em dados não usados no ajuste; realimentação de previsões; rejeição de medições futuras; causalidade da entrada; normalização apenas no treino; reprodução dos dados e separação de identificadores.

## Limitações

Planta linear, simulada e sem ruído. Não há avaliação neural, planta não linear, dados reais, controle em malha fechada ou medição de execução em tempo real.

## Evidências no repositório

Os arquivos completos ficam em `resultados/linear-v1/`, fora da pasta do vault. Os links a seguir funcionam na navegação do repositório; para abrir os arquivos técnicos no computador, use essa pasta.

- [Resumo](../../resultados/linear-v1/resumo.md).
- [Métricas de teste](../../resultados/linear-v1/metricas-teste.csv).
- [Configuração](../../resultados/linear-v1/config.json).
- [Modelo identificado](../../resultados/linear-v1/modelo-arx.json).
- [Registro de versões e hashes](../../resultados/linear-v1/execucao.json).
- [Código preservado da execução](../../resultados/linear-v1/fonte/experimento_linear.py).

Próximo experimento: [[E002 - Mamba-3]].

