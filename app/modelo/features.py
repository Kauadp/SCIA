import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.sparse import csr_matrix, hstack


# ============================================================
# CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

TFIDF_PATH = (
    RAIZ
    / "artifacts"
    / "tfidf"
    / "tfidf_vectorizer.joblib"
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

EMBEDDING_MODEL_NAME = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_DIM = 384
TFIDF_DIM = 100000

DIA_INICIO_COLUMNS = [
    "dia_inicio_Quarta-feira",
    "dia_inicio_Quinta-feira",
    "dia_inicio_Segunda-feira",
    "dia_inicio_Sexta-feira",
    "dia_inicio_Sábado",
    "dia_inicio_Terça-feira",
]

DIA_FIM_COLUMNS = [
    "dia_fim_Quarta-feira",
    "dia_fim_Quinta-feira",
    "dia_fim_Segunda-feira",
    "dia_fim_Sexta-feira",
    "dia_fim_Sábado",
    "dia_fim_Terça-feira",
]

DENSE_DIM = (
    4       # horas
    + 1     # log_duracao
    + 6     # dia inicio
    + 6     # dia fim
    + 3     # estilometria
    + 384   # embedding
)

TOTAL_FEATURES = DENSE_DIM + TFIDF_DIM


# ============================================================
# MODELO DE EMBEDDING
# ============================================================

embedding_model = None


def obter_embedding_model():
    global embedding_model

    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return embedding_model


# ============================================================
# ARTEFATOS
# ============================================================

tfidf_vectorizer = joblib.load(
    TFIDF_PATH
)


# ============================================================
# FEATURES TEMPORAIS
# ============================================================

def criar_features_temporais(df):

    hora_inicio = df["hora_inicio"].to_numpy()
    hora_fim = df["hora_fim"].to_numpy()

    hora_inicio_sin = np.sin(
        2 * np.pi * hora_inicio / 24
    ).reshape(-1, 1)

    hora_inicio_cos = np.cos(
        2 * np.pi * hora_inicio / 24
    ).reshape(-1, 1)

    hora_fim_sin = np.sin(
        2 * np.pi * hora_fim / 24
    ).reshape(-1, 1)

    hora_fim_cos = np.cos(
        2 * np.pi * hora_fim / 24
    ).reshape(-1, 1)

    return (
        hora_inicio_sin,
        hora_inicio_cos,
        hora_fim_sin,
        hora_fim_cos,
    )


# ============================================================
# DUMMIES DOS DIAS
# ============================================================

def criar_dummies_dia(df):

    dummies_inicio = pd.get_dummies(
        df["dia_semana_inicio"],
        prefix="dia_inicio",
        drop_first=True,
        dtype=float
    )

    dummies_inicio = dummies_inicio.reindex(
        columns=DIA_INICIO_COLUMNS,
        fill_value=0
    )

    dummies_fim = pd.get_dummies(
        df["dia_semana_fim"],
        prefix="dia_fim",
        drop_first=True,
        dtype=float
    )

    dummies_fim = dummies_fim.reindex(
        columns=DIA_FIM_COLUMNS,
        fill_value=0
    )

    return (
        dummies_inicio.to_numpy(),
        dummies_fim.to_numpy()
    )


# ============================================================
# FEATURES COMPORTAMENTAIS
# ============================================================

def criar_features_comportamentais(df):

    return np.column_stack([
        df["qtd_palavras"].to_numpy(),
        df["qtd_caracteres"].to_numpy(),
        df["caracter_por_palavra"].to_numpy(),
    ]).astype(np.float32)


# ============================================================
# DURAÇÃO
# ============================================================

def criar_feature_duracao(df):

    return (
        df["log_duracao"]
        .to_numpy()
        .reshape(-1, 1)
        .astype(np.float32)
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def gerar_embeddings(df):

    embeddings = obter_embedding_model().encode(
        df["mensagem"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Dimensão inesperada dos embeddings: "
            f"{embeddings.shape[1]}. "
            f"Esperado: {EMBEDDING_DIM}."
        )

    return embeddings


# ============================================================
# TF-IDF
# ============================================================

def gerar_tfidf(df):

    tfidf = tfidf_vectorizer.transform(
        df["mensagem"].tolist()
    )

    if tfidf.shape[1] != TFIDF_DIM:
        raise ValueError(
            f"Número inesperado de features TF-IDF: "
            f"{tfidf.shape[1]}. "
            f"Esperado: {TFIDF_DIM}."
        )

    return tfidf


# ============================================================
# PIPELINE COMPLETO
# ============================================================

def criar_features(
    df,
    retornar_metadados=False
):

    # --------------------------------------------------------
    # Temporais
    # --------------------------------------------------------

    (
        hora_inicio_sin,
        hora_inicio_cos,
        hora_fim_sin,
        hora_fim_cos
    ) = criar_features_temporais(df)

    # --------------------------------------------------------
    # Dummies dos dias
    # --------------------------------------------------------

    (
        dia_inicio_dummies,
        dia_fim_dummies
    ) = criar_dummies_dia(df)

    # --------------------------------------------------------
    # Duração
    # --------------------------------------------------------

    log_duracao = criar_feature_duracao(
        df
    )

    # --------------------------------------------------------
    # Comportamentais
    # --------------------------------------------------------

    comportamentais = (
        criar_features_comportamentais(df)
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = gerar_embeddings(
        df
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf = gerar_tfidf(
        df
    )

    # --------------------------------------------------------
    # Features densas
    #
    # Ordem EXATA da V2:
    #
    # hora_inicio_sin
    # hora_inicio_cos
    # hora_fim_sin
    # hora_fim_cos
    # log_duracao
    # dia_inicio_dummies
    # dia_fim_dummies
    # qtd_palavras
    # qtd_caracteres
    # caracter_por_palavra
    # embedding
    # --------------------------------------------------------

    features_densas = np.hstack([
        hora_inicio_sin,
        hora_inicio_cos,
        hora_fim_sin,
        hora_fim_cos,
        log_duracao,
        dia_inicio_dummies,
        dia_fim_dummies,
        comportamentais,
        embeddings,
    ]).astype(np.float32)

    # --------------------------------------------------------
    # Validação das features densas
    # --------------------------------------------------------

    if features_densas.shape[1] != DENSE_DIM:

        raise ValueError(
            f"Número inesperado de features densas: "
            f"{features_densas.shape[1]}. "
            f"Esperado: {DENSE_DIM}."
        )

    # --------------------------------------------------------
    # Combinação final
    # --------------------------------------------------------

    X = hstack([
        csr_matrix(features_densas),
        tfidf
    ]).tocsr()

    # --------------------------------------------------------
    # Validação final
    # --------------------------------------------------------

    if X.shape[1] != TOTAL_FEATURES:

        raise ValueError(
            f"Número final de features: "
            f"{X.shape[1]}. "
            f"Esperado: {TOTAL_FEATURES}."
        )

    # --------------------------------------------------------
    # Metadados
    # --------------------------------------------------------

    if not retornar_metadados:

        return X

    metadados = []

    for i in range(len(df)):

        metadados.append({
            "dia_inicio": df.iloc[i][
                "dia_semana_inicio"
            ],

            "dia_fim": df.iloc[i][
                "dia_semana_fim"
            ],

            "hora_inicio": int(
                df.iloc[i]["hora_inicio"]
            ),

            "hora_fim": int(
                df.iloc[i]["hora_fim"]
            ),

            "log_duracao": float(
                df.iloc[i]["log_duracao"]
            ),

            "qtd_palavras": float(
                df.iloc[i]["qtd_palavras"]
            ),

            "qtd_caracteres": float(
                df.iloc[i]["qtd_caracteres"]
            ),

            "caracter_por_palavra": float(
                df.iloc[i]["caracter_por_palavra"]
            )
        })

    return X, metadados

if __name__ == "__main__":

    from preprocessamento import preprocessar
    from app.banco.conexao import db

    df = db.carregar_mensagens_raw()
    df = df.head(100)
    df_clean = preprocessar(df)

    X, metadados = criar_features(
        df_clean,
        retornar_metadados=True
    )

    print("Shape:", X.shape)
    print("Densidade:", X.nnz / (X.shape[0] * X.shape[1]))
    print("Primeiro metadado:")
    print(metadados[0])