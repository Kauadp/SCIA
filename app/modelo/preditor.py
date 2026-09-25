import numpy as np
import joblib

from pathlib import Path

from .preprocessamento import preprocessar
from .features import criar_features
from .modelo import prever


# ============================================================
# CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]

LABEL_ENCODER_PATH = (
    RAIZ
    / "artifacts"
    / "models"
    / "label_encoder.joblib"
)


# ============================================================
# LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

MAPA_NOMES_MODELO = {
    "Você": "Kauã",
    "Jão": "João",
    "Peter Lee": "Peterle",
    "Sr. Quick Açaí": "Quick Açaí",
    "Yuras": "Yuri",
    "Cauã": "Cauã",
    "Arthur": "Arthur",
    "Luquete": "Lucas",
    "André": "André",
    "Luiz": "Luiz",
    "Caio": "Caio",
    "Guido": "Guido",
    "Ítalo": "Ítalo"
}

# ============================================================
# PREDICTOR
# ============================================================

def prever_autor(df_raw):
    """
    Recebe mensagens no formato bruto do banco,
    executa todo o pipeline de preprocessing/features
    e retorna as probabilidades do modelo.
    """

    # --------------------------------------------------------
    # Preprocessamento
    # --------------------------------------------------------

    df_clean = preprocessar(
        df_raw
    )

    if df_clean.empty:
        return None

    print("\n===== DEBUG PREPROCESSAMENTO =====")
    print(df_clean[[
        "membro",
        "dia",
        "hora",
        "caracter_por_mensagem"
    ]])
    print("==================================\n")

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X, metadados = criar_features(
        df_clean,
        retornar_metadados=True
    )

    print("\n===== DEBUG METADADOS =====")
    print(metadados)
    print("===========================\n")

    # --------------------------------------------------------
    # Predição
    # --------------------------------------------------------

    probabilidades = prever(
        X
    )

    # --------------------------------------------------------
    # Classe mais provável
    # --------------------------------------------------------

    indice = np.argmax(
        probabilidades,
        axis=1
    )

    nomes_modelo = label_encoder.inverse_transform(indice)
    nomes = [
        MAPA_NOMES_MODELO.get(nome, nome)
        for nome in nomes_modelo
    ]

    confiancas = probabilidades[
        np.arange(len(indice)),
        indice
    ]

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    resultados = []

    for i in range(len(indice)):

        probabilidades_membros = {
            MAPA_NOMES_MODELO.get(membro, membro): float(probabilidade)
            for membro, probabilidade in zip(
                label_encoder.classes_,
                probabilidades[i]
            )
        }

        resultados.append({
            "autor_previsto": nomes[i],
            "probabilidade": float(
                confiancas[i]
            ),
            "probabilidades": (
                probabilidades_membros
            ),
            "cluster": metadados[i]["cluster"], 
            "dia": metadados[i]["dia"], 
            "hora": metadados[i]["hora"], 
            "caracter_por_mensagem": ( 
                metadados[i]["caracter_por_mensagem"]
            ),
        })

    return resultados

def prever_mensagem(mensagem):
    """
    Prevê o autor de uma única mensagem.

    Parâmetro:
        mensagem: dict no formato produzido pelo parser.
    """

    import pandas as pd

    df_raw = pd.DataFrame([
        mensagem
    ])

    resultados = prever_autor(
        df_raw
    )

    if not resultados:
        return None

    return resultados[0]

