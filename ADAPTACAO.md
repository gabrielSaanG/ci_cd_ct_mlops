# Adaptação do projeto para o HTRU2

Este documento registra a adaptação do projeto de demonstração CI/CD/CT, que vinha com o dataset Breast Cancer do
scikit-learn, para o dataset **HTRU2** (classificação de candidatos a pulsar). O pipeline, os workflows e a
estrutura de pastas foram mantidos: mudaram o domínio, o esquema dos dados, o contrato de qualidade e os testes.

Documento completo: seções 1 a 3 descrevem a adaptação, a seção 4 reúne as evidências E1–E5 executadas no
GitHub Actions e a seção 5 registra a divisão do trabalho.

---

## 1. Dataset escolhido

| Item | Valor |
|---|---|
| Nome | **HTRU2** (High Time Resolution Universe Survey, pesquisa 1) |
| Fonte | UCI Machine Learning Repository, id 372 — <https://archive.ics.uci.edu/dataset/372/htru2> |
| Licença | **CC BY 4.0** (permite uso e redistribuição com atribuição) |
| Tarefa | classificação **binária** |
| Linhas | **17.898** |
| Atributos | **8 colunas numéricas** (nenhuma categórica, nenhum valor ausente no dataset original) |
| Alvo | `alvo` — `1` = pulsar (**1.639 linhas, 9,16%**), `0` = ruído/RFI (16.259 linhas) |

### Por que este dataset

- **Mesmo formato do original**: tabular, binário, só atributos numéricos. Isso deixa o pipeline (DVC, portão de
  qualidade, API) intacto e concentra a adaptação onde interessa — esquema, contrato e testes.
- **Desbalanceado de verdade** (9,16% da classe positiva). Isso obriga a usar recall e precisão da classe positiva no
  contrato, em vez de acurácia, e dá sentido ao `class_weight` e ao cenário 5.
- **Domínio com custo de erro assimétrico**: deixar passar um pulsar (falso negativo) é pior do que marcar ruído como
  candidato (falso positivo), que só gera revisão humana. Isso justifica um recall mínimo alto com precisão mínima
  mais baixa.
- **Licença aberta e tamanho pequeno** (~1,5 MB), então os dados podem ficar versionados no Git, como no projeto do
  professor.

### Domínio em uma frase

Um pulsar é uma estrela de nêutrons que gira rápido e emite um feixe de rádio periódico. Radiotelescópios geram
milhares de *candidatos* por varredura e quase todos são interferência de radiofrequência (RFI). O modelo faz a
triagem automática; o que ele marca como pulsar vai para conferência humana.

### Colunas

As 8 features são estatísticas de duas curvas de cada candidato. Os nomes originais do UCI são descritivos
("Mean of the integrated profile"), então foram renomeados para `snake_case` curto e estável:

| Coluna | Origem (HTRU2) |
|---|---|
| `profile_mean` | média do perfil integrado do pulso |
| `profile_stdev` | desvio-padrão do perfil integrado |
| `profile_skewness` | assimetria do perfil integrado |
| `profile_kurtosis` | curtose (excesso) do perfil integrado |
| `dm_mean` | média da curva DM-SNR |
| `dm_stdev` | desvio-padrão da curva DM-SNR |
| `dm_skewness` | assimetria da curva DM-SNR |
| `dm_kurtosis` | curtose (excesso) da curva DM-SNR |
| `alvo` | `class` do HTRU2 (1 = pulsar) |

> Atenção, uma pegadinha do arquivo original: no `HTRU_2.csv` a ordem das colunas é
> média, desvio, **curtose**, **assimetria** — não média, desvio, assimetria, curtose. O
> `scripts/preparar_htru2.py` lê na ordem real e reordena para a ordem acima, que é a usada em todo o projeto.

### Divisão dos dados

Feita por `python -m mlops_demo.data base`, **estratificada pelo alvo**, **sem sobreposição** e com
**semente fixa 42** (`SEMENTE_DADOS` em `src/mlops_demo/data.py`):

| Conjunto | Arquivo | Linhas | Pulsares | Papel |
|---|---|---|---|---|
| Treino | `data/raw/base_treino.csv` | 12.000 | 1.099 (**9,16%**) | treina o modelo |
| Holdout | `data/raw/base_holdout.csv` | 3.000 | 275 (**9,17%**) | referência fixa de avaliação |
| Reserva | não versionada (recalculada) | 2.898 | 265 (**9,14%**) | origem dos lotes novos de `data/raw/lotes/` |

A proporção da classe positiva se mantém nos três conjuntos (9,16% no dataset completo), e
`tests/test_dados.py::test_nao_ha_vazamento_entre_treino_e_holdout` garante que nenhuma linha aparece em dois
conjuntos. `data/raw/lotes/` começa vazia (só com `.gitkeep`): os lotes são gerados na hora pelos cenários.

---

## 2. O que mudou, arquivo por arquivo

| Arquivo | Mudança | Etapas |
|---|---|---|
| `scripts/preparar_htru2.py` | **novo**: baixa o HTRU2 do UCI, reordena as colunas e grava `data/external/htru2.csv` | 1 |
| `data/external/htru2.csv` | **novo**: dataset completo (17.898 linhas) | 1 |
| `data/raw/base_treino.csv`, `base_holdout.csv` | regerados do HTRU2, divisão estratificada com semente 42 | 1–2 |
| `src/mlops_demo/data.py` | saiu `load_breast_cancer`; entrou a leitura do CSV completo, a divisão em treino/holdout/reserva, `aplicar_drift` nas colunas de escala e a validação com as regras do HTRU2 | 3, 7 |
| `src/mlops_demo/config.py` | `FEATURES` com as 8 colunas, `ALVO = "alvo"`, `CLASSE_MALIGNA` → **`CLASSE_PULSAR = 1`**, `COLUNAS_TAMANHO` → **`COLUNAS_ESCALA`**, limiares novos, `MIN_FEATURES_DRIFT = 2` | 4 |
| `params.yaml` | `class_weight: balanced` para compensar o desbalanceamento (sem isso o recall fica ~0,82, abaixo do contrato) | 5 |
| `src/mlops_demo/features.py`, `train.py` | passaram a usar `FEATURES`/`ALVO` do config (8 colunas) | 5 |
| `src/mlops_demo/evaluate.py` | métricas `recall_pulsar` e `precisao_pulsar` com `pos_label=CLASSE_PULSAR` | 6 |
| `src/mlops_demo/monitor.py` | drift por KS nas 8 features, gatilho em 2 atributos ou mais | 7 |
| `src/mlops_demo/api.py` | payload de **8** atributos (era 30); resposta `classe` = `pulsar`/`não pulsar` e `probabilidade_pulsar` | 8 |
| `demo/smoke_test.py` | remove a coluna `alvo` (era `target`) e confere a resposta no domínio de pulsar | 8 |
| `tests/` | os 4 arquivos adaptados ao novo esquema, mais **1 teste novo por camada** (código, dados, modelo, API) específico do HTRU2 | 9 |
| `demo/cenarios.py` | os 5 cenários reescritos com as colunas do HTRU2, mantendo a intenção pedagógica de cada um | 10 |
| `demo/gerar_lote.py`, `.github/PULL_REQUEST_TEMPLATE.md` | textos do domínio (observações de candidatos, classe pulsar) | 10, 12 |
| `models/champion/` | campeão inicial retreinado no HTRU2 (`python -m mlops_demo.train --campeao`) | 11 |
| `README.md` | problema, dataset (fonte, licença, tamanho), contrato e roteiro de CI/CT/CD reescritos | 12 |

### Campeão inicial

`python -m mlops_demo.train --campeao` treina em `base_treino.csv` e avalia em `base_holdout.csv`:

```json
{
  "recall_pulsar": 0.9345454545454546,
  "precisao_pulsar": 0.8107255520504731,
  "acuracia": 0.974,
  "n_amostras": 3000,
  "origem": "base_treino.csv"
}
```

Recall 0,935 e precisão 0,811 cumprem o contrato com folga, e a acurácia (0,974) fica acima do baseline ingênuo de
"nunca é pulsar" (0,908) — comparação feita em `tests/test_modelo.py::test_modelo_supera_o_baseline_ingenuo`.

### Cenários de prática, reescritos

| N | Mudança | Resultado esperado | Verificado localmente |
|---|---|---|---|
| 1 | `separar_xy` passa a descartar `dm_kurtosis` | CI falha em **testes de código** (7 de 8 colunas) | 2 testes falham em `test_codigo.py` |
| 2 | 5 nulos em `profile_mean` e 3 valores negativos em `dm_mean` | CI falha ao **validar os dados brutos**, antes do treino | `data validar` sai com código 1; 2 testes falham em `test_dados.py` |
| 3 | `C: 0.00005` (regularização forte) | CI falha em **testes de modelo**: recall do pulsar cai para 0,815 (< 0,85) | falha em `test_treino_com_o_codigo_e_params_atuais_cumpre_o_contrato` |
| 4 | lote de 120 linhas com escala do sinal 25% menor | **CI verde** (dados válidos) e **CT dispara**: drift em 4 de 8 atributos, retreina e o portão aprova | CI 7/7 ok; drift 4/8; portão APROVADO |
| 5 | `class_weight: {0: 1, 1: 500}` | recall ~0,996 com precisão 0,132 (< 0,70): CI e portão **reprovam** | 4 testes falham em `test_modelo.py` |

O cenário 5 trocou de sentido em relação ao original: no Breast Cancer a classe de interesse era a `0`, no HTRU2 é a
`1`, então o peso extremo agora vai para a classe `1`.

---

## 3. Contrato de qualidade, limiares e drift

### Limiares (`src/mlops_demo/config.py`)

| Constante | Valor | Por quê |
|---|---|---|
| `RECALL_MINIMO` | **0,85** | perder um pulsar é o erro mais caro; é o número que o contrato protege |
| `PRECISAO_MINIMA` | **0,70** | impede o modelo trivial que chuta "pulsar" para todo mundo (recall 1,0 com precisão ~0,09) |
| `TOLERANCIA_CAMPEAO` | 0,0 | o candidato não pode ter recall menor que o campeão, medido **no mesmo conjunto** |
| `PISO_SANIDADE` | 0,80 | o campeão versionado não pode colapsar no holdout fixo |
| `LIMITE_DRIFT_P` | 0,01 | p-valor do teste KS para marcar um atributo como deslocado |
| `MIN_FEATURES_DRIFT` | **2** | gatilho do CT: drift em 2 dos 8 atributos já dispara o retreino |

**Por que não acurácia.** Com 9,16% de pulsares, prever "não pulsar" para todo mundo dá ~90,8% de acurácia. A
acurácia aparece nos relatórios como referência, nunca no contrato.

**Por que 0,85 e 0,70 (e não 0,90/0,90, como no projeto original).** Os limiares do Breast Cancer eram altos porque
aquele dataset é quase balanceado e mais fácil. Aqui a classe positiva é rara, e com regressão logística +
`balanced` o modelo fica em recall ~0,93 / precisão ~0,81. Os limiares foram postos **abaixo do desempenho atual,
mas acima do que um modelo quebrado alcança**: um recall de 0,82 (o que acontece sem `class_weight`) ou uma precisão
de 0,13 (cenário 5) reprovam.

### Validação dos dados (`validar_dataframe`)

As regras vêm da **definição de cada atributo**, não de cortes empíricos no dataset:

| Regra | Colunas | Motivo |
|---|---|---|
| colunas esperadas, na ordem | todas | `separar_xy` e a API dependem da ordem |
| tipo numérico, sem nulos, sem infinitos | todas | o `StandardScaler` quebra com nulo |
| `alvo` apenas em {0, 1} | `alvo` | problema binário |
| valores **>= 0** | `profile_mean`, `dm_mean` | são médias de grandezas não negativas |
| valores **> 0** | `profile_stdev`, `dm_stdev` | desvio-padrão <= 0 é impossível |
| sem teto e sem piso | `*_skewness`, `*_kurtosis` | assimetria e curtose podem ser negativas (ex.: `dm_kurtosis` chega a −3,14) |

**Não há limite superior de propósito.** Um lote com drift de escala tem de ser considerado *válido* pela validação e
*deslocado* pelo monitor — são duas perguntas diferentes. É isso que faz o cenário 4 ficar com CI verde e CT
disparado.

### Drift

`aplicar_drift` multiplica as colunas de `COLUNAS_ESCALA` por um fator, simulando um **radiotelescópio
recalibrado**: a escala do sinal muda sem que o fenômeno físico mude.

```python
COLUNAS_ESCALA = ["profile_mean", "profile_stdev", "dm_mean", "dm_stdev"]
```

São as 4 colunas de média e desvio — as únicas sensíveis a escala. Assimetria e curtose são adimensionais e quase não
se mexem com multiplicação, então não servem para simular nem detectar esse tipo de drift.

O monitor roda um teste **KS de duas amostras** por atributo, comparando os lotes de `data/raw/lotes/` com
`base_treino.csv`. Resultado medido com o lote do cenário 4 (120 linhas, fator 0,75):

| Atributo | KS | p-valor |
|---|---|---|
| `profile_stdev` | 0,746 | 9,9e-69 |
| `profile_mean` | 0,641 | 4,4e-48 |
| `dm_stdev` | 0,289 | 2,8e-09 |
| `dm_mean` | 0,163 | 3,1e-03 |

**4 de 8 atributos** com drift, acima do gatilho de 2 → CT dispara. `MIN_FEATURES_DRIFT` caiu de 3 para 2 porque o
projeto tem 8 atributos (eram 30): exigir 3 de 8 seria um gatilho bem mais rígido, proporcionalmente, do que 3 de 30.

---

## 4. Evidências no GitHub Actions (E1–E5)

Todas as evidências foram geradas no repositório, com a `main` já adaptada ao HTRU2. Cada cenário virou uma branch
e um Pull Request de demonstração — **nenhum deles deve ser mergeado**.

| Evidência | O que mostra | Cenário / PR | Run | Resultado |
|---|---|---|---|---|
| **E1** | CI verde na `main` adaptada | merge do PR #5 | [CI #11](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933035778) | **sucesso**, 3 jobs |
| **E2** | CI vermelho por bug de código | 1 / [#6](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/6) | [CI #12](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933287280) | **falha** em `Testes de código` |
| **E3** | CI vermelho por dados inválidos | 2 / [#7](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/7) | [CI #13](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933346167) | **falha** em `Validar os dados brutos` |
| **E4** | CI vermelho por contrato de modelo | 3 / [#8](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/8) | [CI #14](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933389115) | **falha** em `Testes de modelo` (recall) |
| **E4b** | o outro extremo do mesmo contrato | 5 / [#10](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/10) | [CI #16](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933490517) | **falha** em `Testes de modelo` (precisão) |
| **E5** | CT: drift, retreino e portão de qualidade | 4 / [#9](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/9) | [CT #5](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933444021) + [CI #15](https://github.com/gabrielSaanG/ci_cd_ct_mlops/actions/runs/45933443898) | **CT disparado e portão aprovado**, com o CI verde |

### E1 — CI verde na `main` adaptada

Run do push de merge do PR #5. Os três jobs passam: `Lint e formatação (ruff)`, `Testes (Python 3.11)` e
`Testes (Python 3.12)`. É a linha de base: a partir daqui, todo vermelho que aparecer foi causado pelo cenário.

> O run do merge do PR #4 (CI #10) aparece como **cancelado**: o merge do #5 veio logo em seguida e o
> `concurrency` do `ci.yml` (`cancel-in-progress: true`) derruba o run anterior do mesmo ref. É comportamento
> esperado do workflow, não uma falha.

### E2 — bug de código

O job de lint passa (o código continua válido e formatado) e os dois jobs de teste falham com exit code 1, no passo
`Testes de código`, antes de treinar qualquer coisa:

```
tests/test_codigo.py::test_separar_xy_devolve_oito_features_e_alvo
tests/test_codigo.py::test_separar_xy_impoe_a_ordem_das_colunas
```

Um bug de contrato de esquema é pego em segundos, sem gastar treino.

### E3 — dados inválidos

A falha acontece no passo `Validar os dados brutos`, que roda **antes** de todos os testes. Saída do run:

```
base_treino.csv: FALHOU: valores nulos: {'profile_mean': 5}; valores negativos em ['dm_mean'] (médias devem ser >= 0)
base_holdout.csv: OK
Error: Process completed with exit code 1.
```

Os passos seguintes (`Testes de código`, `de dados`, `de modelo`, `de API`) aparecem com 0s: foram pulados. Dado
ruim é barrado na porta de entrada.

### E4 — contrato de modelo

Aqui nada está quebrado: o código é válido, os dados são válidos, o modelo treina sem erro. O que falha é o
**contrato**, no passo `Testes de modelo`:

```
AssertionError: {'recall_pulsar': 0.8145454545454546, 'precisao_pulsar': 0.8517110266159695,
                 'acuracia': 0.97, 'n_amostras': 3000}
assert 0.8145454545454546 >= 0.85
```

Repare na acurácia: **0,97**, praticamente igual à do campeão (0,974). E a precisão até *subiu*. Se o contrato
fosse acurácia, essa regressão passaria batido. É a justificativa prática dos limiares escolhidos na seção 3.

A **E4b** mostra o extremo oposto, com `class_weight: {0: 1, 1: 500}`: recall 0,996 e precisão 0,132. Recall quase
perfeito e modelo inútil — de cada 100 candidatos marcados como pulsar, 87 são ruído. É o piso de precisão fazendo
o trabalho dele.

### E5 — treinamento contínuo

O lote com drift deixa o **CI verde** (CI #15): os dados são válidos, só estão deslocados. Quem reage é o **CT**,
e o `Summary` do run CT #5 traz o relatório completo:

```
Drift: 4 de 8 atributos
Lotes comparados: lote_01.csv (120 linhas)

| Atributo      | KS    | p-valor |
| profile_stdev | 0.746 | 9.9e-69 |
| profile_mean  | 0.641 | 4.4e-48 |
| dm_stdev      | 0.289 | 2.8e-09 |
| dm_mean       | 0.163 | 3.1e-03 |

Gatilho (>= 2 atributos com drift): DISPARADO, retreinar
```

O CT então roda `dvc repro` e submete o candidato ao portão:

| Modelo | Recall pulsar | Precisão pulsar | Amostras |
|---|---|---|---|
| campeao | 0,937 | 0,816 | 3060 |
| candidato | 0,937 | 0,806 | 3060 |

| Regra | Resultado | Valor |
|---|---|---|
| recall pulsar do candidato >= 0,85 | passou | 0,9366 |
| precisão pulsar do candidato >= 0,70 | passou | 0,8061 |
| recall pulsar do candidato >= recall do campeão | passou | 0,9366 x 0,9366 |

**Portão: APROVADO.** Os números do run batem exatamente com os medidos em `python demo/local.py ct` na máquina
local, o que é o efeito esperado da semente fixa.

Em evento de `pull_request` o CT **não promove o candidato e não abre o PR de campeão** — os dois passos têm a
guarda `github.event_name != 'pull_request'`. A promoção só acontece depois do merge na `main`. Por isso estes PRs
de evidência podem ficar abertos (ou ser fechados) sem efeito nenhum sobre o modelo em produção.

Os cenários 3 e 5 também disparam o workflow CT, porque mexem em `params.yaml` (CT #4 e CT #6). Os dois terminam
com sucesso e sem retreinar: não há lote novo em `data/raw/lotes/`, então o gatilho de drift não dispara e o
workflow encerra com "Gatilho não disparado: nada a fazer". É o comportamento correto — CT por evento só retreina
quando os dados mudam.

## 5. Divisão do trabalho

| Etapas | Responsável | PR |
|---|---|---|
| 1–2 — dados HTRU2, download e divisão treino/holdout/reserva | André | [#1](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/1) |
| 3–4 — contrato do modelo e validação de dados | Matheus | [#2](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/2) |
| 5–7 — modelo, avaliação na classe pulsar e gatilho de drift | Débora | [#3](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/3) |
| 8–9 — API de 8 features e testes das 4 camadas | Wesley | [#4](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/4) |
| 10–12 — cenários, campeão inicial e documentação | João | [#5](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/5) |
| Evidências E1–E5 e seções 4–5 deste documento | João | [#6](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/6) a [#10](https://github.com/gabrielSaanG/ci_cd_ct_mlops/pull/10) |

### Como reproduzir tudo localmente

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev,ct]"
python demo/local.py ci          # 7/7 ok
python demo/cenarios.py listar   # os 5 cenários
```

Para repetir qualquer evidência: `git switch -c cenario-N main`, `python demo/cenarios.py aplicar N`, commit e
push. O `demo/local.py ci` reproduz o CI inteiro sem internet, e o `demo/local.py ct` reproduz o CT.
