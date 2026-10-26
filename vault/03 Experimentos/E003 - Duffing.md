---
tipo: experimento
id: E003
status: executado-localmente
data_execucao: 2026-10-26
atualizado: 2026-10-26
tags: [tcc, experimento, duffing, narx, mamba]
---

# E003 — oscilador de Duffing

**Resultado:** o ARX deixa de ser exato (8,9 cm em 50 passos). O NARX polinomial de grau 3 com quatro atrasos aproxima o sistema quase até a precisão numérica (5,5 × 10⁻⁹ m), mas não o contém exatamente (veja Interpretação). A rede Mamba-3 fica entre os dois: (1,53 ± 0,61) cm, sem nenhuma informação sobre a forma da não linearidade. Execução de 26/10/2026, com apoio do assistente; o grupo ainda deve revisar.

- **Pergunta:** quando o sistema sai da estrutura do ARX, onde fica a rede em relação a um método tradicional linear e a um não linear?
- **Protocolo de referência:** [[Protocolo comum]], com as mesmas divisões, origens, horizontes e métricas.
- **Mudança em relação ao E001/E002:**
  - Planta $m\ddot p + c\dot p + kp + k_3p^3 = u$, com $k_3 = 20$ N/m³ e os demais parâmetros iguais. Integração RK4 com 10 subpassos por período, sob ZOH.
  - Força com amplitude sorteada em [−2, 2] N a cada patamar (APRBS), em vez de ±1 N.
  - Nova referência: NARX polinomial.
- **Não linearidade:** nas posições do treino, a força cúbica tem RMS 1,08 vez o da parte linear e chega a 3,6 vezes na maior elongação (0,85 m).
- **Dados:** semente `20261026`; 20/5/5 trajetórias de 2.048 passos; sem ruído.
- **Referências clássicas:**
  - ARX com ordens em {2, 4, 8}².
  - NARX polinomial: todos os monômios até o grau g de y[t..t−n+1] e u[t..t−n+1], com n ∈ {2, 3, 4} e g ∈ {2, 3}, ajustado por mínimos quadrados do erro de um passo.
  - Ambos escolhidos pela simulação de 50 passos na validação.
- **Rede:** a mesma do [[E002 - Mamba-3]], com a mesma grade de taxas, as mesmas sementes e o mesmo critério. Selecionada lr3e-3; a média da lr1e-2 ficou praticamente empatada.
- **Pastas:** `resultados/duffing-v1/` (dados e referências clássicas) e `resultados/mamba3-duffing-v1/` (rede).

## Resultados no teste

RMSE da trajetória simulada [m], 545 origens:

| Modelo | Parâmetros | 1 passo | 10 passos | 50 passos |
|---|---:|---:|---:|---:|
| Persistência | 0 | 3,47 × 10⁻² | 2,04 × 10⁻¹ | 3,80 × 10⁻¹ |
| ARX(8,8) | 17 | 1,11 × 10⁻⁵ | 8,37 × 10⁻³ | 8,92 × 10⁻² |
| NARX(4,4) grau 3 | 165 | 4,3 × 10⁻¹⁰ | 2,0 × 10⁻⁹ | 5,5 × 10⁻⁹ |
| Mamba-3 (média ± desvio, 3 sementes) | 34.949 | (0,89 ± 0,25) × 10⁻³ | (6,07 ± 1,88) × 10⁻³ | (1,53 ± 0,61) × 10⁻² |

## Interpretação e limites

- **Correção de 26/10 (revisão):** o NARX não é exato "porque a mola é cúbica e cabe na classe". O mapa amostrado de uma EDO com mola cúbica não é um polinômio finito das entradas e saídas passadas: o NARX(2,2) de grau 3, que seria a "estrutura física", erra 1,35 × 10⁻⁴ m na validação. O erro de ~10⁻⁹ m só aparece com quatro atrasos, numa regressão mal condicionada. A nota de leitura gravada em `resultados/mamba3-duffing-v1/resumo.md` repete essa afirmação imprecisa. Ela não foi editada, para preservar o registro da execução.
- Na validação, dois dos seis candidatos NARX divergiram na simulação e foram descartados.
- O escolhido tem posto incompleto (137 de 165 colunas). Isso vem em boa parte da excitação: com patamares de pelo menos 5 amostras, vários produtos de entradas atrasadas são sempre nulos (só os monômios da entrada têm posto 22 de 35). O sinal não excita persistentemente esses regressores, e nenhum modelo foi testado com entradas que mudam a cada amostra.
- A grade favoreceu o NARX: o grau máximo (3) foi escolhido sabendo que a planta é cúbica.
- O ARX(8,8) está na borda da grade. Num diagnóstico posterior (`resultados/diagnostico-e004-v1/`), ordens até 64 mantiveram o erro em ~8,8 cm: a limitação é estrutural.
- Em um passo, o ARX(8,8) erra menos que a rede, mas em 50 passos erra seis vezes mais. É o caso mais claro, até agora, de por que avaliar só um passo engana. Veja [[Previsao e simulacao]].
- Em termos relativos, o NRMSE da rede em 50 passos é 5,4%, contra 2,4% na planta linear. Comparar só os centímetros entre plantas engana, porque o desvio da posição é 0,28 m aqui e 0,47 m na planta linear.
- Por semente, a rede erra de 4 a 8 vezes menos que o ARX em 50 passos.
- Em relação ao E002 mudaram juntos a planta, a excitação (APRBS, ±2 N) e a semente: as diferenças não podem ser atribuídas só à não linearidade.
- Os melhores checkpoints ficaram perto da época 300, de 300: a rede provavelmente está subtreinada.
- Ver [[E004 - Duffing com ruido]] para o que acontece quando a hipótese de dados limpos cai.
- Três sementes; dispersão maior que no E002 (1,1 a 2,2 cm em 50 passos).
- A excitação também mudou em relação ao E001. As plantas linear e de Duffing não são comparadas diretamente entre si.

Veja [[Decisao 2026-10 - Tese identificacao vs tradicionais]] e [[Relatorio 2]].
