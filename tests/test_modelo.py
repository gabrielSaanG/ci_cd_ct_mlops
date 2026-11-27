"""Testes de modelo: limiares, baseline e comportamento do HTRU2."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from mlops_demo import config
from mlops_demo.evaluate import avaliar, calcular_metricas
from mlops_demo.features import separar_xy
from mlops_demo.train import treinar

pytestmark = pytest.mark.modelo


def test_campeao_versionado_foi_certificado_pelo_portao():
    with open(config.METRICAS_CAMPEAO, encoding="utf-8") as arquivo:
        declaradas = json.load(arquivo)
    assert declaradas["recall_pulsar"] >= config.RECALL_MINIMO, declaradas
    assert declaradas["precisao_pulsar"] >= config.PRECISAO_MINIMA, declaradas


def test_campeao_versionado_nao_colapsou_na_base_de_referencia(campeao, base_holdout):
    metricas = avaliar(campeao, base_holdout)
    assert metricas["recall_pulsar"] >= config.PISO_SANIDADE, metricas
    assert metricas["precisao_pulsar"] >= config.PISO_SANIDADE, metricas


def test_treino_com_o_codigo_e_params_atuais_cumpre_o_contrato(modelo_atual, base_holdout):
    metricas = avaliar(modelo_atual, base_holdout)
    assert metricas["recall_pulsar"] >= config.RECALL_MINIMO, metricas
    assert metricas["precisao_pulsar"] >= config.PRECISAO_MINIMA, metricas


def test_modelo_supera_o_baseline_ingenuo(modelo_atual, base_treino, base_holdout):
    X_tr, y_tr = separar_xy(base_treino)
    X_te, y_te = separar_xy(base_holdout)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    nota_baseline = calcular_metricas(y_te, baseline.predict(X_te))["acuracia"]
    nota_modelo = calcular_metricas(y_te, modelo_atual.predict(X_te))["acuracia"]
    assert nota_modelo > nota_baseline + 0.05


def test_treino_e_deterministico(base_treino, params, xy_holdout):
    X_te, _ = xy_holdout
    a = treinar(base_treino, params).predict_proba(X_te)
    b = treinar(base_treino, params).predict_proba(X_te)
    assert np.allclose(a, b)


def test_modelo_consegue_memorizar_um_lote_pequeno(base_treino, params):
    pequeno = base_treino.sample(40, random_state=0)
    modelo = treinar(pequeno, params)
    X, y = separar_xy(pequeno)
    assert (modelo.predict(X) == y).mean() >= 0.90


def test_probabilidades_sao_validas(modelo_atual, xy_holdout):
    X_te, _ = xy_holdout
    probas = modelo_atual.predict_proba(X_te)
    assert np.all((probas >= 0) & (probas <= 1))
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_invariancia_a_ruido_minimo(modelo_atual, xy_holdout):
    X_te, _ = xy_holdout
    rng = np.random.default_rng(0)
    ruidoso = X_te * (1 + rng.normal(0, 1e-4, X_te.shape))
    mudou = (modelo_atual.predict(X_te) != modelo_atual.predict(ruidoso)).mean()
    assert mudou <= 0.02


def test_direcional_escala_do_sinal_nao_reduz_chance_de_pulsar(modelo_atual, xy_holdout):
    """Aumentar as colunas de escala do sinal não reduz a probabilidade média de pulsar."""
    X_te, _ = xy_holdout
    coluna = list(modelo_atual.classes_).index(config.CLASSE_PULSAR)
    maior = X_te.copy()
    for c in config.COLUNAS_ESCALA:
        maior[c] = maior[c] * 1.10
    antes = modelo_atual.predict_proba(X_te)[:, coluna].mean()
    depois = modelo_atual.predict_proba(maior)[:, coluna].mean()
    assert depois >= antes


def test_htru2_modelo_aponta_pulsar_para_a_amostra_mais_confiante(modelo_atual, base_holdout):
    X, _ = separar_xy(base_holdout)
    coluna = list(modelo_atual.classes_).index(config.CLASSE_PULSAR)
    assert (
        modelo_atual.predict(X.iloc[[modelo_atual.predict_proba(X)[:, coluna].argmax()]])[0]
        == config.CLASSE_PULSAR
    )
