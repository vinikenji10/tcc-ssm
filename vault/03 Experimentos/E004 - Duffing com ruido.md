---
tipo: experimento
id: E004
status: executado-localmente
data_execucao: 2026-10-26
atualizado: 2026-10-26
tags: [tcc, experimento, duffing, ruido, narx, mamba]
---

# E004 — Duffing com ruído de medição

**Resultado:** na execução registrada, com ruído de 2 cm, o NARX(4,4) passa a errar 14,5 cm em 50 passos e o ARX 10,2 cm. A rede fica em (1,54 ± 0,28) cm, medido contra a posição sem ruído. **A vantagem da rede é artefato da grade:** num diagnóstico posterior, o NARX com 10 atrasos chega a 1,9 cm, equivalente à rede. Com 12 atrasos, ele seria escolhido pela validação e diverge no teste. Execução de 26/10/2026, com apoio do assistente; o grupo ainda deve revisar.

- **Pergunta:** o que acontece com cada família de modelos quando a medição tem ruído?
- **Mudança em relação ao E003:** só o ruído. Gaussiano branco, com desvio de 0,02 m, somado à posição medida em todos os conjuntos.
  - Mesma semente, mesmas forças e mesmas posições verdadeiras do E003.
  - O desvio da posição é de cerca de 0,28 m, uma relação sinal-ruído de cerca de 14 em amplitude.
- **O que os modelos veem:** só a medição ruidosa. Ajuste, normalização, escolha na validação e condições iniciais de cada simulação usam a posição com ruído.
- **Como o erro é medido:** no teste, as mesmas previsões são comparadas também com a posição sem ruído. Ela fica em `diagnostico/saida-sem-ruido.npz`, fora dos dados dos modelos.
- **Escolhas na validação:** ARX(8,8), NARX(4,4) de grau 3 e rede com lr3e-3. São as mesmas estruturas do E003.
- **Pastas:** `resultados/duffing-ruido-v1/` e `resultados/mamba3-duffing-ruido-v1/`.

## Resultados no teste

RMSE da trajetória simulada [m], contra a posição sem ruído. A última coluna é o RMSE em 50 passos contra a medição com ruído, que tem piso de cerca de 0,02 m.

| Modelo | 1 passo | 10 passos | 50 passos | 50 passos, contra a medição |
|---|---:|---:|---:|---:|
| Persistência | 4,06 × 10⁻² | 2,04 × 10⁻¹ | 3,80 × 10⁻¹ | 3,81 × 10⁻¹ |
| ARX(8,8) | 1,84 × 10⁻² | 5,14 × 10⁻² | 1,02 × 10⁻¹ | 1,04 × 10⁻¹ |
| NARX(4,4) grau 3 | 2,46 × 10⁻² | 6,72 × 10⁻² | 1,45 × 10⁻¹ | 1,46 × 10⁻¹ |
| Mamba-3 (3 sementes) | (9,1 ± 0,3) × 10⁻³ | (1,13 ± 0,12) × 10⁻² | (1,54 ± 0,28) × 10⁻² | 2,53 × 10⁻² |

## Diagnóstico posterior (26/10, revisão)

Feito **depois** de conhecer o teste. Serve para entender o resultado, não para escolher modelos. Script: `experimentos/diagnostico_e004.py`; saída: `resultados/diagnostico-e004-v1/diagnostico.json`.

| NARX grau 3 | n=4 | n=5 | n=6 | n=8 | n=10 | n=12 | Mamba-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Termos | 165 | 286 | 455 | 969 | 1.771 | 2.925 | 34.949 parâmetros |
| Validação, contra medição [m] | 0,149 | 0,097 | 0,062 | 0,034 | 0,027 | 0,025 | 0,025 |
| Teste, contra posição sem ruído [m] | 0,145 | 0,094 | 0,059 | 0,029 | 0,019 | diverge | 0,015 |

Com n=14, o NARX diverge já na validação. O ARX não melhora com ordens até 64 (~10 cm).

De onde vem o erro do NARX(4,4) no teste:
- Com ajuste ruidoso e condição inicial ruidosa: 14,5 cm.
- Com o mesmo ajuste e condição inicial limpa: 14,3 cm. O dominante, portanto, é o viés dos coeficientes.
- Com ajuste em dados sem ruído e condição inicial ruidosa: 5,1 cm.

Parte da vantagem da rede vem de estimar o estado inicial com 256 medições. Os modelos tradicionais partem de n medições ruidosas.

## Interpretação e limites

- A queda do NARX(4,4) é o viés conhecido do ajuste por erro de equação com regressores ruidosos. Mais atrasos reduzem o erro, mas o modelo fica numericamente frágil.
- Remédios tradicionais **não usados** aqui: ajuste pelo erro de simulação (*output error*), seleção de poucos termos (FROLS) e modelo de ruído (NARMAX). Implementá-los é a primeira tarefa de novembro (veja [[Cronograma]]).
- O diagnóstico confirma em parte que o aquecimento com 256 medições funciona como filtro: mesmo o NARX ajustado sem ruído erra 5,1 cm partindo de 4 saídas ruidosas.
- O E004 foi planejado depois de ver o teste do E003, sobre as mesmas trajetórias. Para a versão final, reservar um conjunto de teste novo.
- Testamos um único nível de ruído e três sementes.

Veja [[Decisao 2026-10 - Tese identificacao vs tradicionais]] e [[Relatorio 2]].
