"""Testes de contrato da API HTRU2: entrada de oito features e saída de pulsar."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api
cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_devolve_o_formato_combinado(base_holdout):
    exame = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": exame})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_pulsar", "versao_codigo"}
    assert corpo["classe"] in {"pulsar", "não pulsar"}
    assert 0.0 <= corpo["probabilidade_pulsar"] <= 1.0


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


def test_predict_acerta_um_pulsar_e_um_nao_pulsar_claros(base_holdout, campeao):
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_PULSAR)
    p_pulsar = campeao.predict_proba(X)[:, coluna]
    i_pulsar = int(p_pulsar.argmax())
    i_nao_pulsar = int(p_pulsar.argmin())
    for indice, esperado in ((i_pulsar, "pulsar"), (i_nao_pulsar, "não pulsar")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado


def test_api_htru2_rejeita_payload_com_nove_features(base_holdout):
    exame = [*base_holdout[config.FEATURES].iloc[0].tolist(), 0.0]
    resposta = cliente.post("/predict", json={"atributos": exame})
    assert resposta.status_code == 422
