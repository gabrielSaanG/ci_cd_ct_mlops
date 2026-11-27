# Adaptação do projeto CI/CD/CT para o HTRU2

Projeto de demonstração de CI/CD/CT adaptado do dataset Breast Cancer para o **HTRU2**, classificação de
candidatos a pulsar. Pipeline, workflows e estrutura de pastas mantidos; mudaram domínio, esquema, contrato,
testes e cenários.

## 1. Problema e dados

Um **pulsar** é uma estrela de nêutrons que gira rápido e emite um feixe de rádio periódico. Radiotelescópios
registram milhares de *candidatos* por varredura e quase todos são interferência de radiofrequência (RFI). Quem
usaria o modelo: a equipe de astrofísica que faz a triagem — o modelo filtra, a pessoa confere o que sobra.

| Item | Valor |
|---|---|
| Dataset | **HTRU2** (High Time Resolution Universe Survey) |
| Fonte | UCI Machine Learning Repository, id 372 — <https://archive.ics.uci.edu/dataset/372/htru2> |
| Licença | **CC BY 4.0** |
| Tamanho | **17.898 linhas**, **8 atributos numéricos**, nenhum valor ausente |
| Alvo | `alvo`: 1 = pulsar (**1.639 linhas, 9,16%**), 0 = ruído/RFI |

As 8 features são média, desvio-padrão, assimetria e curtose de duas curvas: o perfil integrado do pulso
(`profile_*`) e a curva DM-SNR (`dm_*`). Nenhum dado pessoal ou sensível.

Divisão **estratificada, sem sobreposição, semente fixa 42** (`python -m mlops_demo.data base`):

| Conjunto | Arquivo | Linhas | Pulsares |
|---|---|---|---|
| Treino | `data/raw/base_treino.csv` | 12.000 | 1.099 (9,16%) |
| Holdout | `data/raw/base_holdout.csv` | 3.000 | 275 (9,17%) |
| Reserva | não versionada | 2.898 | 265 (9,14%) |

`tests/test_dados.py::test_nao_ha_vazamento_entre_treino_e_holdout` garante que nenhuma linha se repete.

## 2. Contrato de qualidade

**Classe positiva: pulsar (1).** O erro mais caro é o **falso negativo** — um pulsar descartado some da fila de
revisão e ninguém mais olha. O falso positivo só gera trabalho de conferência humana. Daí recall alto com
precisão mais tolerante.

| Constante | Valor | Justificativa |
|---|---|---|
| `RECALL_MINIMO` | **0,85** | perder um pulsar é o erro caro; é o número que o contrato protege |
| `PRECISAO_MINIMA` | **0,70** | sem ele, chutar "pulsar" para tudo daria recall 1,0 com precisão 0,09 |
| `TOLERANCIA_CAMPEAO` | 0,0 | o candidato não pode ter recall menor que o campeão, no mesmo conjunto |
| `PISO_SANIDADE` | 0,80 | o campeão versionado não pode colapsar no holdout fixo |
| `LIMITE_DRIFT_P` | 0,01 | p-valor do teste KS |
| `MIN_FEATURES_DRIFT` | **2** | eram 3 de 30 atributos; agora 2 de 8, proporção equivalente |

**Acurácia não entra no contrato.** Com 9,16% de pulsares, prever "não pulsar" sempre já dá 90,8%. Os limiares
foram postos abaixo do desempenho atual (recall 0,935 / precisão 0,811) e acima do que um modelo quebrado alcança:
recall 0,82 sem `class_weight`, ou precisão 0,13 com peso extremo, reprovam.

## 3. O que mudou, arquivo por arquivo

| Etapa | Arquivo | Mudança |
|---|---|---|
| 1 | `data/raw/*.csv`, `data/external/htru2.csv`, `scripts/preparar_htru2.py` | dados do HTRU2; script de download reproduz o CSV completo |
| 2 | `src/mlops_demo/data.py` | sai `load_breast_cancer`; `dividir_dataset`/`criar_base` leem o CSV e dividem com semente 42 |
| 3 | `src/mlops_demo/config.py` | `FEATURES` (8 colunas), `ALVO`, `CLASSE_MALIGNA` → **`CLASSE_PULSAR = 1`**, `COLUNAS_TAMANHO` → **`COLUNAS_ESCALA`**, limiares novos |
| 4 | `validar_dataframe` em `data.py` | regras do domínio: médias >= 0, desvios > 0, assimetria/curtose livres, alvo em {0,1} |
| 5 | `features.py`, `train.py`, `params.yaml` | regressão logística mantida; `class_weight: balanced` para o desbalanceamento |
| 6 | `evaluate.py` | `recall_pulsar` e `precisao_pulsar` com `pos_label=CLASSE_PULSAR` |
| 7 | `aplicar_drift` em `data.py`, `monitor.py` | drift de escala nas 4 colunas de média/desvio; `MIN_FEATURES_DRIFT = 2` |
| 8 | `api.py`, `demo/smoke_test.py` | payload de **8** atributos (era 30); resposta `pulsar`/`não pulsar` e `probabilidade_pulsar` |
| 9 | `tests/` | 4 arquivos adaptados + **1 teste novo por camada** específico do HTRU2 |
| 10 | `demo/cenarios.py` | os 5 cenários com as colunas do HTRU2 |
| 11 | `models/champion/` | campeão inicial: recall 0,9345, precisão 0,8107, acurácia 0,974 em 3.000 linhas (depois substituído pelo CT, seção 7) |
| 12 | `README.md` | problema, dataset, contrato e roteiro reescritos |

**Testes novos (etapa 9):** `test_htru2_separar_xy_preserva_o_rotulo_binario_de_pulsar` (código),
`test_htru2_tem_exatamente_oito_atributos_de_observacao` (dados),
`test_htru2_modelo_aponta_pulsar_para_a_amostra_mais_confiante` (modelo),
`test_api_htru2_rejeita_payload_com_nove_features` (API).

**Cenários (etapa 10):** 1 descarta `dm_kurtosis`; 2 corrompe `profile_mean` e `dm_mean`; 3 usa `C: 0.00005`;
4 gera lote com escala do sinal 25% menor; 5 usa `class_weight: {0: 1, 1: 500}`. O cenário 5 inverteu o sentido:
no Breast Cancer a classe de interesse era a `0`, aqui é a `1`.

## 4. As cinco evidências

| | O que mostra | Execução | Resultado |
|---|---|---|---|
| **E1** | CI verde na `main` adaptada | [CI #18](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45934322601) | 3 jobs verdes |
| **E2** | CI vermelho por falha controlada (cenário 1, PR #6) | [CI #12](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933287280) | falha em `Testes de código` |
| **E3** | CT com drift e portão **aprovando** (cenário 4, PR #9) | [CT #5](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933444021) | drift 4/8, portão APROVADO |
| **E4** | CT com portão **reprovando** (PR #12) | [CT #7](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45936115932) | precisão 0,133 < 0,70, portão REPROVADO |
| **E5** | CD verde com a tag `v1.0.0` | [CD #1](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45936146293) | CI, imagem, teste de fumaça e publicação no ghcr.io |

Evidências extras de CI vermelho, nas outras camadas: [CI #13](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933346167)
(dados inválidos, falha em `Validar os dados brutos`), [CI #14](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933389115)
(contrato de modelo, recall 0,8145 < 0,85) e [CI #16](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933490517)
(precisão 0,132 < 0,70).

**E3, em detalhe.** Lote de 120 linhas com as colunas de escala multiplicadas por 0,75. O CI fica **verde** — os
dados são válidos, só deslocados. O CT detecta por KS: `profile_stdev` 0,746, `profile_mean` 0,641, `dm_stdev`
0,289, `dm_mean` 0,163 — **4 de 8**, acima do gatilho de 2. Retreina e o portão aprova (candidato: recall 0,9366,
precisão 0,8061).

**E4, em detalhe.** Mesmo lote com drift, mais `class_weight: {0: 1, 1: 500}`. O CT dispara, retreina e o portão
**reprova**: recall 0,9965 (passa) mas precisão 0,1334 (falha). **Duas das três regras passaram** — inclusive
"candidato melhor que o campeão em recall". Se o portão olhasse só recall, promoveria um modelo em que 87 de cada
100 alertas são ruído. O campeão não muda: o passo de promoção tem `if: success()`.

**E5, em detalhe.** A tag `v1.0.0` disparou o CD, que rodou o CI inteiro, construiu a imagem, subiu o container,
chamou `/health` e `/predict` pelo `demo/smoke_test.py` e só então publicou. Imagem:
[`ghcr.io/gabrielsaang/ci_cd_ct_mlops`](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pkgs/container/ci_cd_ct_mlops),
**pública**, tags `1.0.0`, `latest` e `sha-ac041bd`, para linux/amd64 e arm64. Download sem autenticação
confirmado direto no registro (manifesto HTTP 200 com token anônimo):

```bash
docker run -p 8000:8000 ghcr.io/gabrielsaang/ci_cd_ct_mlops:1.0.0
python demo/smoke_test.py http://localhost:8000
```

## 5. Limites e riscos

**Por que o portão sozinho não garante um bom modelo.**

- Ele mede **duas métricas agregadas em um conjunto fixo**. Um modelo pode cumprir recall e precisão globais e
  ainda assim falhar sistematicamente em uma faixa do espaço de entrada — candidatos de baixa relação sinal-ruído,
  por exemplo. Média alta esconde subgrupo ruim.
- O conjunto de avaliação **envelhece**. Metade de cada lote novo entra na avaliação, mas o holdout original de
  3.000 linhas continua dominando. Depois de muitos ciclos, o portão compararia modelos em dados que não se
  parecem mais com a produção.
- Ele compara candidato e campeão, não candidato e **ótimo**. Nada impede degradação lenta: cada candidato um
  pouco pior que o anterior dentro da tolerância, e em dez ciclos o modelo caiu muito.
- **Drift de rótulo e de conceito passam batido.** O monitor roda KS só nas features. Se a definição operacional
  de "pulsar confirmado" mudar, ou se os rótulos novos vierem enviesados, o portão aprova um modelo que aprendeu a
  coisa errada.
- O teste KS com amostras de 120 linhas é sensível demais a lote pequeno e insensível a drift lento e acumulado.
- Nada aqui mede **latência, custo ou tamanho do artefato** — um modelo aprovado pode ser inviável de servir.

**O que monitoraríamos em produção.**

| O quê | Por quê |
|---|---|
| Distribuição das 8 features e da **probabilidade prevista** | drift de entrada e deslocamento do limiar de decisão |
| Taxa de positivos preditos por dia | salto súbito costuma ser problema de instrumentação, não descoberta |
| **Precisão medida na revisão humana** | é o único recall/precisão real; o do holdout é estimativa |
| Latência e taxa de erro da API, por versão | regressão de serviço, não de modelo |
| Métricas por subgrupo (faixa de `dm_mean`, por telescópio) | achar o subgrupo ruim que a média esconde |
| Defasagem entre previsão e confirmação | mede quanto tempo o modelo roda sem feedback real |

Em produção de verdade, o portão seria um dos portões: somaríamos revisão humana amostrada, *canary release* por
porcentagem de tráfego e rollback automático por métrica de serviço.

## 6. Contribuição e uso de IA

| Integrante | Etapas | PR |
|---|---|---|
| André | 1–2: dados HTRU2, download e divisão | [#1](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/1) |
| Matheus | 3–4: contrato e validação | [#2](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/2) |
| Débora | 5–7: modelo, métricas e drift | [#3](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/3) |
| Wesley | 8–9: API e testes | [#4](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/4) |
| João | 10–12: cenários, campeão e documentação | [#5](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/5) |
| João | evidências E1–E5 e este relatório | [#6](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/6) a [#12](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/12) |

**Uso de IA.** As etapas 10 a 12, as evidências E1–E5 e este relatório foram feitos com apoio do **Claude Code**
(Anthropic), usado para: reescrever `demo/cenarios.py` no domínio do HTRU2, calibrar os valores dos cenários 3 e 5
medindo o efeito real no holdout, redigir o `README.md` e este documento, e operar a geração das evidências no
GitHub Actions. Todos os números citados foram **medidos**, não estimados: cada cenário foi executado localmente
com `python demo/local.py ci` e `ct` antes do push, e conferido depois no log do Actions. A escolha do dataset e
as decisões de contrato (0,85 / 0,70) foram do grupo.

## 7. Desafio opcional: promoção automática

Com `PROMOCAO_AUTOMATICA = true` e a `main` sem proteção, o ciclo roda inteiro sem ninguém no caminho. Um único
`git push` de um lote novo em `data/raw/lotes/` para a `main` produziu, em cerca de 3 minutos:

| Passo | Quem fez | Execução |
|---|---|---|
| Detectar drift, retreinar, aplicar o portão | [CT #8](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45937245358) | drift 4/8, portão APROVADO |
| Publicar o campeão na `main` | `github-actions[bot]` | commit `6eaf7f9` — "ct: promove novo modelo campeão (portão aprovado)" |
| Criar a tag da versão | `github-actions[bot]` | `v1.1.0` |
| Construir, testar e publicar a imagem | [CD #2](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45937320495) | disparado pelo bot, 2m15s |

Resultado: `models/champion/metrics.json` passou a registrar `"origem": "treinamento contínuo (CT)"`, com recall
0,9366 e precisão 0,8061, e o registro ganhou a tag `1.1.0` ao lado da `1.0.0`. **Nenhuma pessoa aprovou nada
depois do push do lote.**

### Vantagens

- **Tempo de resposta.** Entre o dado novo chegar e o modelo atualizado estar servindo em produção passaram-se
  minutos, não dias. Em um domínio onde o instrumento é recalibrado, isso é a diferença entre reagir e acumular
  erro.
- **O critério fica explícito e versionado.** A decisão de promover deixa de ser julgamento de quem está de
  plantão e vira código revisável: os limiares estão no `config.py` e mudá-los exige Pull Request.
- **Elimina o gargalo humano e o erro de operação.** Ninguém esquece de criar a tag, ninguém publica a imagem
  errada, ninguém promove um modelo que não passou no portão.
- **Rastreabilidade.** Cada promoção deixa commit, tag, run e imagem com o mesmo SHA. Dá para reconstruir
  exatamente qual modelo estava no ar em qualquer data.

### Riscos

- **O portão vira o único juízo.** Tudo o que ele não mede passa (ver seção 5): desempenho por subgrupo, drift de
  conceito, latência, custo. Um modelo pode cumprir os dois limiares e ainda ser pior na prática.
- **Dados ruins promovem modelos ruins.** O gatilho é drift, não qualidade. Se um lote chegar corrompido de um
  jeito que a validação não pega — uma unidade trocada, por exemplo — o CT retreina em cima disso e publica.
- **Degradação lenta não é vista.** Cada candidato pode ser um pouco pior que o anterior e ainda passar.
  Em dez ciclos o modelo caiu bastante sem nenhum run vermelho.
- **A `main` precisa ficar desprotegida** para o bot escrever nela. Isso enfraquece o controle para *todo mundo*,
  não só para o robô, e é um preço de segurança real.
- **Sem revisão, não há quem note o estranho.** Uma pessoa revisando um PR de promoção repara em "o recall subiu
  demais de repente". O portão não.
- **Versões proliferam.** Cada execução agendada com drift gera tag e imagem novas. Sem política de limpeza, o
  registro cresce sem controle.

### O que faríamos em produção

Manteríamos a promoção automática apenas com rede de proteção em volta: *canary* com fatia pequena de tráfego
antes dos 100%, rollback automático por métrica de serviço, revisão humana amostrada sobre o que o modelo marca
como pulsar, e proteção da `main` com exceção só para o bot (no GitHub, um *bypass* de ruleset para um App) em vez
de deixar a branch aberta para todos.

**Estado atual do repositório:** a variável continua em `true` para que a evidência seja verificável. Para voltar
ao modo com revisão humana, basta trocá-la para `false` em Settings > Secrets and variables > Actions > Variables.

---

Reproduzir localmente: `pip install -e ".[dev,ct]"`, depois `python demo/local.py ci` (7/7 ok) e
`python demo/cenarios.py listar`.
