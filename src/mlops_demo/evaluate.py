"""Métricas de avaliação. A classe de interesse é o PULSAR (rótulo 1)."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from mlops_demo.config import CLASSE_PULSAR
from mlops_demo.features import separar_xy


def calcular_metricas(y_real, y_previsto) -> dict:
    """Recall e precisão da classe pulsar, mais a acurácia como referência.

    Com ~9% de pulsares, a acurácia engana (prever tudo como "não pulsar" já dá ~91%).
    Por isso o contrato usa recall e precisão da classe positiva.
    """
    return {
        "recall_pulsar": float(recall_score(y_real, y_previsto, pos_label=CLASSE_PULSAR)),
        "precisao_pulsar": float(
            precision_score(y_real, y_previsto, pos_label=CLASSE_PULSAR, zero_division=0)
        ),
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "n_amostras": int(len(y_real)),
    }


def avaliar(modelo, df: pd.DataFrame) -> dict:
    """Avalia um modelo treinado em um DataFrame com as colunas do projeto."""
    X, y = separar_xy(df)
    return calcular_metricas(y, modelo.predict(X))
