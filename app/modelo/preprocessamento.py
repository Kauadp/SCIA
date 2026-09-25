import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))

import re
import pandas as pd

from app.banco.conexao import db


def limpar_para_embedding(texto):
    texto = str(texto).strip().lower()

    # Remove apenas artefatos do WhatsApp / sistema
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)

    texto = re.sub(
        r"\b(unknown|null|omitid[oa]s?|ocultad[oa]s?|"
        r"imagem|figurinha|audio|áudio|video|vídeo|"
        r"mensagem|apagad[oa]s?|media|anexo|voz|"
        r"documento|contato|editad[oa]s?)\b",
        "",
        texto
    )

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


map_dias = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo"
}


dias_ordenados = [
    "Segunda-feira",
    "Terça-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sábado",
    "Domingo",
]


def preprocessar(df):
    df = df.copy()

    df["mensagem_embedding"] = df["mensagem"].apply(
        limpar_para_embedding
    )

    df = df[
        df["mensagem_embedding"].str.len() > 0
    ].reset_index(drop=True)

    df["qtd_caracteres"] = (
        df["mensagem"]
        .astype(str)
        .str.len()
        .astype(float)
    )

    df["qtd_palavras"] = (
        df["mensagem"]
        .astype(str)
        .apply(lambda x: len(x.split()))
        .astype(float)
    )

    df["caracter_por_mensagem"] = (
        df["qtd_caracteres"] /
        df["qtd_palavras"].clip(lower=1)
    )

    df["data_hora"] = pd.to_datetime(df["data_hora"], utc=True)
    df["data_hora"] = df["data_hora"].dt.tz_convert("America/Sao_Paulo")

    df["dia"] = (
        df["data_hora"]
        .dt.dayofweek
        .map(map_dias)
    )

    df["hora"] = df["data_hora"].dt.hour

    return df[
        [
            "membro",
            "mensagem_embedding",
            "dia",
            "hora",
            "caracter_por_mensagem"
        ]
    ]


if __name__ == "__main__":
    df = db.carregar_mensagens_raw()

    df_clean = preprocessar(df)

    print(df_clean.head())