import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))

import re
import numpy as np
import pandas as pd

from app.banco.conexao import db


MAP_DIAS = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo"
}


DIAS_ORDENADOS = [
    "Segunda-feira",
    "Terça-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sábado",
    "Domingo",
]


def limpar_para_embedding(texto):
    texto = str(texto).strip().lower()

    # Remove URLs
    texto = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        texto
    )

    # Remove apenas artefatos do WhatsApp / sistema
    texto = re.sub(
        r"\b(unknown|null|omitid[oa]s?|ocultad[oa]s?|"
        r"imagem|figurinha|audio|áudio|video|vídeo|"
        r"mensagem|apagad[oa]s?|media|anexo|voz|"
        r"documento|contato|editad[oa]s?)\b",
        "",
        texto
    )

    # Normaliza espaços
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def preprocessar(df):
    df = df.copy()

    # ---------------------------------------------------------
    # 1. Limpeza textual
    # ---------------------------------------------------------

    df["mensagem"] = df["mensagem"].apply(
        limpar_para_embedding
    )

    # Remove mensagens que ficaram vazias após a limpeza
    df = df[
        df["mensagem"].str.len() > 0
    ].copy()

    # ---------------------------------------------------------
    # 2. Data/hora
    # ---------------------------------------------------------

    df["data_hora"] = pd.to_datetime(
        df["data_hora"],
        utc=True
    )

    df["data_hora"] = (
        df["data_hora"]
        .dt.tz_convert("America/Sao_Paulo")
    )

    # Ordenação cronológica dentro de cada membro
    df = df.sort_values(
        ["membro", "data_hora"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Agrupamento de 5 mensagens por membro
    # ---------------------------------------------------------

    df["grupo_5"] = (
        df.groupby("membro")
        .cumcount()
        .floordiv(5)
    )

    df_consolidado = (
        df.groupby(
            ["membro", "grupo_5"],
            as_index=False
        )
        .agg(
            data_inicio=("data_hora", "min"),
            data_fim=("data_hora", "max"),
            mensagem=("mensagem", " ".join),
            total_msgs=("mensagem", "count")
        )
    )

    # Mantém somente grupos completos de 5 mensagens
    df_consolidado = df_consolidado[
        df_consolidado["total_msgs"] == 5
    ].copy()

    # ---------------------------------------------------------
    # 4. Features temporais
    # ---------------------------------------------------------

    df_consolidado["dia_semana_inicio"] = (
        df_consolidado["data_inicio"]
        .dt.dayofweek
        .map(MAP_DIAS)
    )

    df_consolidado["dia_semana_fim"] = (
        df_consolidado["data_fim"]
        .dt.dayofweek
        .map(MAP_DIAS)
    )

    df_consolidado["hora_inicio"] = (
        df_consolidado["data_inicio"]
        .dt.hour
    )

    df_consolidado["hora_fim"] = (
        df_consolidado["data_fim"]
        .dt.hour
    )

    # ---------------------------------------------------------
    # 5. Duração da rajada
    # ---------------------------------------------------------

    duracao_minutos = (
        (
            df_consolidado["data_fim"]
            - df_consolidado["data_inicio"]
        )
        .dt.total_seconds()
        / 60
    )

    df_consolidado["log_duracao"] = np.log1p(
        duracao_minutos
    )

    # ---------------------------------------------------------
    # 6. Features comportamentais / estilométricas
    # ---------------------------------------------------------

    df_consolidado["qtd_caracteres"] = (
        df_consolidado["mensagem"]
        .str.len()
        .astype(float)
    )

    df_consolidado["qtd_palavras"] = (
        df_consolidado["mensagem"]
        .str.split()
        .str.len()
        .astype(float)
    )

    df_consolidado["caracter_por_palavra"] = (
        df_consolidado["qtd_caracteres"]
        / df_consolidado["qtd_palavras"].clip(lower=1)
    )

    # ---------------------------------------------------------
    # 7. Retorno final da V2
    # ---------------------------------------------------------

    return df_consolidado[
        [
            "membro",
            "mensagem",
            "log_duracao",
            "dia_semana_inicio",
            "dia_semana_fim",
            "hora_inicio",
            "hora_fim",
            "qtd_caracteres",
            "qtd_palavras",
            "caracter_por_palavra"
        ]
    ].reset_index(drop=True)


if __name__ == "__main__":
    df = db.carregar_mensagens_raw()

    df_clean = preprocessar(df)

    print(df_clean.head())
    print()
    print(f"Grupos gerados: {len(df_clean)}")
    print()
    print(df_clean["membro"].value_counts())