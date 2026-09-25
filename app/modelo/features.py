import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

UMAP_PATH = (
    RAIZ
    / "artifacts"
    / "clustering"
    / "umap_cluster.joblib"
)

KMEANS_PATH = (
    RAIZ
    / "artifacts"
    / "clustering"
    / "kmeans_25.joblib"
)

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


DIA_COLUMNS = [
    "dia_Quarta-feira",
    "dia_Quinta-feira",
    "dia_Segunda-feira",
    "dia_Sexta-feira",
    "dia_Sábado",
    "dia_Terça-feira",
]


CLUSTER_25_COLUMNS = [
    "cluster_25_1",
    "cluster_25_2",
    "cluster_25_3",
    "cluster_25_4",
    "cluster_25_5",
    "cluster_25_6",
    "cluster_25_7",
    "cluster_25_8",
    "cluster_25_9",
    "cluster_25_10",
    "cluster_25_11",
    "cluster_25_12",
    "cluster_25_13",
    "cluster_25_14",
    "cluster_25_15",
    "cluster_25_16",
    "cluster_25_17",
    "cluster_25_18",
    "cluster_25_19",
    "cluster_25_20",
    "cluster_25_21",
    "cluster_25_22",
    "cluster_25_23",
    "cluster_25_24",
]


# ============================================================
# MODELO DE EMBEDDING
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# CARREGAMENTO DOS ARTEFATOS
# ============================================================

umap_model = joblib.load(
    UMAP_PATH
)

kmeans_model = joblib.load(
    KMEANS_PATH
)

tfidf_vectorizer = joblib.load(
    TFIDF_PATH
)


# ============================================================
# FEATURES TEMPORAIS
# ============================================================

def criar_features_temporais(df):
    hora = df["hora"].to_numpy()

    hora_sin = np.sin(
        2 * np.pi * hora / 24
    ).reshape(-1, 1)

    hora_cos = np.cos(
        2 * np.pi * hora / 24
    ).reshape(-1, 1)

    return hora_sin, hora_cos


# ============================================================
# DUMMIES DO DIA
# ============================================================

def criar_dummies_dia(df):
    dummies = pd.get_dummies(
        df["dia"],
        prefix="dia",
        drop_first=True,
        dtype=float
    )

    dummies = dummies.reindex(
        columns=DIA_COLUMNS,
        fill_value=0
    )

    return dummies.to_numpy()


# ============================================================
# ESTILOMETRIA
# ============================================================

def criar_feature_estilometria(df):
    return (
        df["caracter_por_mensagem"]
        .to_numpy()
        .reshape(-1, 1)
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def gerar_embeddings(df):
    embeddings = embedding_model.encode(
        df["mensagem_embedding"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.astype(np.float32)


# ============================================================
# CLUSTERING
# ============================================================

def criar_clusters(embeddings):
    embedding_reduzido = umap_model.transform(
        embeddings
    )

    clusters = kmeans_model.predict(
        embedding_reduzido
    )

    return clusters


# ============================================================
# DUMMIES DOS CLUSTERS
# ============================================================

def criar_dummies_cluster25(clusters):
    dummies = pd.get_dummies(
        pd.Series(clusters),
        prefix="cluster_25",
        drop_first=True,
        dtype=float
    )

    dummies = dummies.reindex(
        columns=CLUSTER_25_COLUMNS,
        fill_value=0
    )

    return dummies.to_numpy()


# ============================================================
# TF-IDF
# ============================================================

def gerar_tfidf(df):
    return tfidf_vectorizer.transform(
        df["mensagem_embedding"]
    )


# ============================================================
# PIPELINE COMPLETO DE FEATURES
# ============================================================

def criar_features(
    df,
    retornar_metadados=False
):
    """
    Transforma o dataframe pré-processado
    nas features utilizadas pelo MLP.

    Ordem:

    1. hora_sin
    2. hora_cos
    3. dia_dummies
    4. caracter_por_mensagem
    5. cluster_25_dummies
    6. embedding
    7. tfidf

    Se retornar_metadados=True, retorna também
    as features interpretáveis utilizadas no processamento.
    """

    # ----------------------------
    # Temporais
    # ----------------------------

    hora_sin, hora_cos = (
        criar_features_temporais(df)
    )

    # ----------------------------
    # Dia
    # ----------------------------

    dia_dummies = criar_dummies_dia(df)

    # ----------------------------
    # Estilometria
    # ----------------------------

    estilometria = (
        criar_feature_estilometria(df)
    )

    # ----------------------------
    # Embedding
    # ----------------------------

    embeddings = gerar_embeddings(df)

    # ----------------------------
    # Clusters
    # ----------------------------

    clusters = criar_clusters(
        embeddings
    )

    cluster_dummies = (
        criar_dummies_cluster25(
            clusters
        )
    )

    # ----------------------------
    # TF-IDF
    # ----------------------------

    tfidf = gerar_tfidf(df)

    # ----------------------------
    # Features densas
    # ----------------------------

    features_densas = np.hstack([
        hora_sin,
        hora_cos,
        dia_dummies,
        estilometria,
        cluster_dummies,
        embeddings,
    ])

    # ----------------------------
    # Validação
    # ----------------------------

    if features_densas.shape[1] != 417:
        raise ValueError(
            f"Número inesperado de features densas: "
            f"{features_densas.shape[1]}. "
            f"Esperado: 417."
        )

    if tfidf.shape[1] != 100000:
        raise ValueError(
            f"Número inesperado de features TF-IDF: "
            f"{tfidf.shape[1]}. "
            f"Esperado: 100000."
        )

    # ----------------------------
    # Combinação final
    # ----------------------------

    from scipy.sparse import csr_matrix, hstack

    X = hstack([
        csr_matrix(features_densas),
        tfidf
    ]).tocsr()

    # ----------------------------
    # Validação final
    # ----------------------------

    if X.shape[1] != 100417:
        raise ValueError(
            f"Número final de features: "
            f"{X.shape[1]}. "
            f"Esperado: 100417."
        )

    # ----------------------------
    # Retorno
    # ----------------------------

    if not retornar_metadados:
        return X

    metadados = []

    for i in range(len(df)):

        metadados.append({
            "cluster": int(clusters[i]),
            "dia": df.iloc[i]["dia"],
            "hora": int(df.iloc[i]["hora"]),
            "caracter_por_mensagem": float(
                df.iloc[i]["caracter_por_mensagem"]
            )
        })

    return X, metadados