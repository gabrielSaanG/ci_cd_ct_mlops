"""Configuração central do projeto: caminhos, nomes de colunas e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# dados
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"
TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# esquema dos dados
FEATURES = [
    "profile_mean",
    "profile_stdev",
    "profile_skewness",
    "profile_kurtosis",
    "dm_mean",
    "dm_stdev",
    "dm_skewness",
    "dm_kurtosis",
]
ALVO = "alvo"
# HTRU2: 1 = PULSAR (classe positiva, ~9% das linhas), 0 = ruído/RFI.
CLASSE_PULSAR = 1
# Atributos sensíveis à escala do sinal (média e desvio do perfil integrado e da curva DM-SNR).
# Usados para simular drift (recalibração do radiotelescópio) e no teste direcional.
COLUNAS_ESCALA = ["profile_mean", "profile_stdev", "dm_mean", "dm_stdev"]

# limiares de qualidade (o "contrato" do modelo). Mudar aqui exige revisão em PR.
# A precisão mínima evita o modelo trivial que marca tudo como pulsar (recall 1,0, precisão ~0,09).
RECALL_MINIMO = 0.85
PRECISAO_MINIMA = 0.70
TOLERANCIA_CAMPEAO = 0.0  # o candidato não pode ter recall menor que o campeão
PISO_SANIDADE = 0.80  # o campeão não pode colapsar na base de referência (holdout fixo)
LIMITE_DRIFT_P = 0.01  # p-valor do teste KS
MIN_FEATURES_DRIFT = 3  # gatilho do CT: quantos atributos com drift disparam o retreino


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
