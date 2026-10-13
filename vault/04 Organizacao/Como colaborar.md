---
tipo: guia
status: instrucoes
atualizado: 2026-10-12
tags: [tcc, grupo, github, obsidian]
---

# Como colaborar

O repositório guarda código, dados e resultados; o vault resume o contexto para o grupo. As notas não substituem a configuração e as métricas originais de cada execução.

## Obter o projeto

Repositório do projeto: [vinikenji10/tcc-ssm](https://github.com/vinikenji10/tcc-ssm).

O repositório é público por solicitação do usuário: qualquer pessoa pode consultar e clonar o projeto. Para enviar diretamente branches ao repositório, o proprietário precisa adicionar colaboradores. Também é possível contribuir por fork e pull request. Convites ainda não foram enviados por esta tarefa.

```bash
git clone https://github.com/vinikenji10/tcc-ssm.git
cd tcc-ssm
```

No Obsidian, selecione **Abrir pasta como cofre/vault** e escolha `tcc-ssm/vault`. Leia [[Inicio]]. A configuração usa recursos nativos, sem plugins comunitários.

## Antes de editar

Com o diretório de trabalho limpo, sincronize a cópia:

```bash
git pull --ff-only
git switch -c nome-curto-da-atividade
```

Se houver mudanças locais, revise-as antes de atualizar; não descarte o trabalho de outro integrante para resolver conflitos.

## Para código ou experimentos

1. Descrever a pergunta e a mudança de método.
2. Executar as verificações apropriadas.
3. Usar uma nova pasta de resultados, preservando execuções anteriores.
4. Registrar no vault a configuração, resultado, limites e arquivos de evidência.
5. Enviar as alterações para revisão do grupo.

Exemplo, adaptando os caminhos à alteração feita:

```bash
git add experimentos/ vault/
git commit -m "Descreve a alteração realizada"
git push -u origin nome-curto-da-atividade
```

Abra um pull request no GitHub para o grupo revisar. A integração pode ser combinada pela equipe; não foi definida uma política obrigatória de revisão nesta tarefa.

## Para notas

- Usar [[Registro de experimento]], [[Reuniao]] ou [[Decisao]] como ponto de partida.
- Indicar data, fonte e situação: planejado, em execução, executado ou bloqueado.
- Relacionar a nota a [[Estado atual]], [[Tarefas]] e ao experimento correspondente.
- Ao mudar uma métrica ou figura, vincular a execução de origem e atualizar a cópia em `Anexos` quando necessário.

## Arquivos locais

O ambiente Python, caches e o layout pessoal de janelas do Obsidian ficam fora do Git. São compartilhados os arquivos de configuração do vault que ajudam a navegação do grupo.

## Evitar perda de contexto

Mantenham resultados medidos, hipóteses e propostas identificados. Mamba-3 só entra como resultado quando for executado no nosso protocolo; resultados do artigo permanecem literatura de apoio.
