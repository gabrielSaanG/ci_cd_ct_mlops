"""Baixa o HTRU2 (UCI, CC BY 4.0) e salva o dataset completo em data/external/htru2.csv.

O split treino/holdout/reserva é feito por mlops_demo.data (semente fixa).
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"

# Ordem REAL do arquivo HTRU_2.csv (sem cabeçalho)
COLUNAS_ORIGINAIS = [
    "profile_mean",
    "profile_stdev",
    "profile_kurtosis",
    "profile_skewness",
    "dm_mean",
    "dm_stdev",
    "dm_kurtosis",
    "dm_skewness",
    "alvo",
]
# Ordem de saída combinada com o grupo
COLUNAS_SAIDA = [
    "profile_mean",
    "profile_stdev",
    "profile_skewness",
    "profile_kurtosis",
    "dm_mean",
    "dm_stdev",
    "dm_skewness",
    "dm_kurtosis",
    "alvo",
]

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "external" / "htru2.csv"


def main() -> None:
    with urllib.request.urlopen(URL) as resp:
        dados = resp.read()
    with zipfile.ZipFile(io.BytesIO(dados)) as z:
        nome = next(n for n in z.namelist() if n.endswith("HTRU_2.csv"))
        with z.open(nome) as f:
            df = pd.read_csv(f, header=None, names=COLUNAS_ORIGINAIS)
    df = df[COLUNAS_SAIDA]
    assert len(df) == 17898 and df["alvo"].isin([0, 1]).all()

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DESTINO, index=False)
    print(f"salvo em {DESTINO}: {len(df)} linhas | pulsares: {int(df['alvo'].sum())}")


if __name__ == "__main__":
    main()
