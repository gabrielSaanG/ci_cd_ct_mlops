"""Cenários de prática: cada um aplica uma mudança para você ver o pipeline reagir.

Domínio: classificação de candidatos a pulsar (HTRU2). Os cenários mexem nas colunas
reais do dataset (perfil integrado e curva DM-SNR), não em atributos de outro problema.

Uso (a partir da raiz do projeto, sempre em uma branch nova):
    python demo/cenarios.py listar
    python demo/cenarios.py aplicar 1

Depois de aplicar: git add -A, git commit, git push e abra o Pull Request.
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mlops_demo.data import gerar_lote  # noqa: E402

DESCRICAO = {
    1: (
        "Bug no pré-processamento: descarta dm_kurtosis, a última das 8 features. "
        "Esperado: CI falha em test_codigo (separar_xy devolve 7 colunas)."
    ),
    2: (
        "Dados corrompidos na base de treino: nulos em profile_mean e dm_mean negativo. "
        "Esperado: CI falha ao validar os dados brutos e em test_dados."
    ),
    3: (
        "Hiperparâmetro ruim (regularização forte, C muito pequeno). "
        "Esperado: CI falha em test_modelo (recall do pulsar ~0,81 < 0,85)."
    ),
    4: (
        "Lote novo com drift: radiotelescópio recalibrado, escala do sinal 25% menor. "
        "Esperado: CI verde e CT dispara, retreina e o portão aprova."
    ),
    5: (
        "class_weight extremo a favor do pulsar. "
        "Esperado: recall ~1,0 com precisão ~0,13; CI e portão do CT reprovam."
    ),
}


def _substituir(arquivo: Path, antigo: str, novo: str) -> None:
    texto = arquivo.read_text(encoding="utf-8")
    if antigo not in texto:
        raise SystemExit(f"Trecho não encontrado em {arquivo.name}. O cenário já foi aplicado?")
    arquivo.write_text(texto.replace(antigo, novo, 1), encoding="utf-8")
    print(f"alterado: {arquivo.relative_to(RAIZ)}")


def cenario_1() -> None:
    """Quebra o contrato de esquema: o pipeline passa a treinar com 7 das 8 features."""
    _substituir(
        RAIZ / "src" / "mlops_demo" / "features.py",
        "X = df[FEATURES].copy()",
        "X = df[FEATURES[:-1]].copy()",
    )


def cenario_2() -> None:
    """Corrompe a base de treino com valores impossíveis para o domínio do HTRU2."""
    caminho = RAIZ / "data" / "raw" / "base_treino.csv"
    df = pd.read_csv(caminho)
    df.loc[0:4, "profile_mean"] = None  # média do perfil integrado ausente
    df.loc[5:7, "dm_mean"] = -10.0  # média da curva DM-SNR não pode ser negativa
    df.to_csv(caminho, index=False)
    print(
        "alterado: data/raw/base_treino.csv "
        "(5 nulos em profile_mean, 3 valores negativos em dm_mean)"
    )


def cenario_3() -> None:
    """Regularização forte demais: o modelo deixa de alcançar o recall do contrato."""
    _substituir(RAIZ / "params.yaml", "C: 1.0", "C: 0.00005")


def cenario_4() -> None:
    """Lote com drift de escala nas colunas monitoradas (COLUNAS_ESCALA)."""
    caminho = gerar_lote("lote_01", n=120, fator_drift=0.75, semente=1)
    print(
        "criado:",
        Path(caminho).relative_to(RAIZ),
        "(profile_mean, profile_stdev, dm_mean e dm_stdev 25% menores)",
    )


def cenario_5() -> None:
    """Peso extremo na classe minoritária: marca quase tudo como pulsar."""
    _substituir(RAIZ / "params.yaml", "class_weight: balanced", "class_weight: {0: 1, 1: 500}")


CENARIOS = {1: cenario_1, 2: cenario_2, 3: cenario_3, 4: cenario_4, 5: cenario_5}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "listar":
        for numero, texto in DESCRICAO.items():
            print(f"{numero}. {texto}")
    elif len(sys.argv) == 3 and sys.argv[1] == "aplicar" and sys.argv[2].isdigit():
        numero = int(sys.argv[2])
        if numero not in CENARIOS:
            raise SystemExit("Cenário inexistente. Use: python demo/cenarios.py listar")
        print(f"Cenário {numero}: {DESCRICAO[numero]}")
        CENARIOS[numero]()
    else:
        print(__doc__)
        sys.exit(2)
