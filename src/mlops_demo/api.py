"""API de inferência (FastAPI) servindo o modelo campeão HTRU2.

Local:   uvicorn mlops_demo.api:app --reload
Docker:  docker run -p 8000:8000 mlops-demo
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops_demo import __version__, config

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")

app = FastAPI(title="Classificação de pulsares (demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class Exame(BaseModel):
    """As oito features do HTRU2, na mesma ordem do treino."""

    atributos: list[float] = Field(min_length=8, max_length=8)


class Predicao(BaseModel):
    classe: str
    probabilidade_pulsar: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.post("/predict", response_model=Predicao)
def prever(exame: Exame) -> Predicao:
    entrada = pd.DataFrame([exame.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    coluna = list(_modelo.classes_).index(config.CLASSE_PULSAR)
    p_pulsar = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="pulsar" if previsto == config.CLASSE_PULSAR else "não pulsar",
        probabilidade_pulsar=round(p_pulsar, 4),
        versao_codigo=__version__,
    )
