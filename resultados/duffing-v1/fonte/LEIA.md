# Sobre esta cópia do código

`experimento_duffing.py` ganhou, depois desta execução, o trecho que mede o erro contra a posição sem ruído (usado no E004) e a função `clean_metrics`. Por isso o hash em `execucao.json` difere desta cópia. Para dados sem ruído esse trecho não é executado: em 26/10/2026 esta versão foi rodada de novo com `config-duffing.json` e reproduziu os mesmos dados (mesmo SHA-256 de `dados.npz`) e o mesmo `metricas-teste.csv`, byte a byte.
