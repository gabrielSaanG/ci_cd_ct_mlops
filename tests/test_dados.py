"""Testes de dados: esquema, valores e vazamento no HTRU2."""

import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import listar_lotes, validar_dataframe

pytestmark = pytest.mark.dados


def _arquivos_de_dados():
    return [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]


@pytest.mark.parametrize("caminho", _arquivos_de_dados(), ids=lambda c: c.name)
def test_arquivo_respeita_esquema_e_valores(caminho):
    problemas = validar_dataframe(pd.read_csv(caminho))
    assert problemas == [], f"{caminho.name}: {problemas}"


def test_base_de_treino_tem_as_duas_classes_em_proporcao_razoavel(base_treino):
    proporcao_pulsar = (base_treino[config.ALVO] == config.CLASSE_PULSAR).mean()
    assert 0.07 <= proporcao_pulsar <= 0.12


def test_nao_ha_vazamento_entre_treino_e_holdout(base_treino, base_holdout):
    chaves_treino = set(map(tuple, base_treino[config.FEATURES].round(6).to_numpy()))
    chaves_holdout = set(map(tuple, base_holdout[config.FEATURES].round(6).to_numpy()))
    assert chaves_treino.isdisjoint(chaves_holdout)


def test_nao_ha_linhas_duplicadas_na_base_de_treino(base_treino):
    assert not base_treino.duplicated().any()


def test_validador_detecta_nulos_e_faixas_invalidas(base_treino):
    corrompido = base_treino.copy()
    corrompido.loc[0, "profile_mean"] = None
    corrompido.loc[1, "dm_mean"] = -5.0
    corrompido.loc[2, "dm_stdev"] = 0.0
    problemas = validar_dataframe(corrompido)
    assert any("nulos" in p for p in problemas)
    assert any("negativos" in p for p in problemas)
    assert any("desvios-padrão" in p for p in problemas)


def test_validador_aceita_assimetria_e_curtose_negativas(base_treino):
    dados = base_treino.copy()
    dados.loc[0, ["profile_skewness", "profile_kurtosis", "dm_skewness", "dm_kurtosis"]] = -1.5
    assert validar_dataframe(dados) == []


def test_validador_detecta_tipo_e_alvo_invalidos(base_treino):
    texto = base_treino.copy()
    texto["profile_mean"] = texto["profile_mean"].astype(str)
    assert any("não numéricas" in p for p in validar_dataframe(texto))
    alvo_ruim = base_treino.copy()
    alvo_ruim.loc[0, config.ALVO] = 2
    assert any("fora de {0, 1}" in p for p in validar_dataframe(alvo_ruim))


def test_validador_detecta_esquema_diferente(base_treino):
    problemas = validar_dataframe(base_treino.drop(columns=["profile_mean"]))
    assert problemas and "faltando=['profile_mean']" in problemas[0]
    fora_de_ordem = base_treino[[*reversed(config.FEATURES), config.ALVO]]
    assert validar_dataframe(fora_de_ordem)


def test_htru2_tem_exatamente_oito_atributos_de_observacao(base_treino):
    assert config.ALVO not in config.FEATURES
    assert len(config.FEATURES) == 8
    assert list(base_treino.columns) == [*config.FEATURES, config.ALVO]
