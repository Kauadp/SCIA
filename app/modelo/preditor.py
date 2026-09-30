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
    Recebe mensagens brutas do banco, executa todo o pipeline
    V2 e retorna uma previsão por grupo de 5 mensagens.
    """

    # --------------------------------------------------------
    # Preprocessamento
    # --------------------------------------------------------

    df_clean = preprocessar(
        df_raw
    )

    if df_clean.empty:
        return None

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X, metadados = criar_features(
        df_clean,
        retornar_metadados=True
    )

    # --------------------------------------------------------
    # Predição
    # --------------------------------------------------------

    probabilidades = prever(
        X
    )

    # --------------------------------------------------------
    # Classe mais provável
    # --------------------------------------------------------

    indices = np.argmax(
        probabilidades,
        axis=1
    )

    nomes_modelo = label_encoder.inverse_transform(
        indices
    )

    nomes = [
        MAPA_NOMES_MODELO.get(
            nome,
            nome
        )
        for nome in nomes_modelo
    ]

    confiancas = probabilidades[
        np.arange(len(indices)),
        indices
    ]

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    resultados = []

    for i in range(len(indices)):

        probabilidades_membros = {
            MAPA_NOMES_MODELO.get(
                membro,
                membro
            ): float(probabilidade)

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

            "dia_inicio": metadados[i]["dia_inicio"],
            "dia_fim": metadados[i]["dia_fim"],

            "hora_inicio": metadados[i]["hora_inicio"],
            "hora_fim": metadados[i]["hora_fim"],

            "log_duracao": metadados[i]["log_duracao"],

            "qtd_palavras": metadados[i]["qtd_palavras"],
            "qtd_caracteres": metadados[i]["qtd_caracteres"],
            "caracter_por_palavra": (
                metadados[i]["caracter_por_palavra"]
            ),
        })

    return resultados

# ============================================================
# TESTE LOCAL
# ============================================================

if __name__ == "__main__":

    from app.banco.conexao import db

    print("\n" + "=" * 60)
    print("TESTE DO PREDITOR V2")
    print("=" * 60)

    df_raw = db.carregar_mensagens_raw()

    print(f"\nMensagens carregadas: {len(df_raw)}")

    df_raw = df_raw.head(100)

    print(f"Mensagens usadas no teste: {len(df_raw)}")

    df_clean = preprocessar(
        df_raw
    )

    if df_clean.empty:

        print("\nNenhum grupo completo encontrado.")
        raise SystemExit

    resultados = prever_autor(
        df_raw
    )

    if resultados is None:

        print("\nNenhum resultado produzido.")
        raise SystemExit

    acertos = 0
    total = len(resultados)

    print(f"\nGrupos previstos: {total}")

    for i, resultado in enumerate(
        resultados,
        start=1
    ):

        membro_real = df_clean.iloc[i - 1]["membro"]

        membro_real = MAPA_NOMES_MODELO.get(
            membro_real,
            membro_real
        )

        autor_previsto = resultado[
            "autor_previsto"
        ]

        acertou = (
            autor_previsto == membro_real
        )

        if acertou:
            acertos += 1

        status = "✓" if acertou else "✗"

        print("\n" + "-" * 60)
        print(f"GRUPO {i}")
        print(f"Real:     {membro_real}")
        print(f"Previsto: {autor_previsto}")
        print(
            f"Confiança: "
            f"{resultado['probabilidade']:.4f}"
        )
        print(f"Resultado: {status}")

        print(
            f"Período: "
            f"{resultado['dia_inicio']} "
            f"{resultado['hora_inicio']}h → "
            f"{resultado['dia_fim']} "
            f"{resultado['hora_fim']}h"
        )

        print(
            f"Log duração: "
            f"{resultado['log_duracao']:.4f}"
        )

        print(
            f"Palavras: "
            f"{resultado['qtd_palavras']:.0f}"
        )

        print(
            f"Caracteres: "
            f"{resultado['qtd_caracteres']:.0f}"
        )

        print(
            f"Caracteres/palavra: "
            f"{resultado['caracter_por_palavra']:.4f}"
        )

        print("\nProbabilidades:")

        probabilidades_ordenadas = sorted(
            resultado["probabilidades"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for membro, probabilidade in (
            probabilidades_ordenadas
        ):
            print(
                f"  {membro:<15} "
                f"{probabilidade:.4f}"
            )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    acuracia = acertos / total

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)

    print(f"Grupos avaliados: {total}")
    print(f"Acertos:          {acertos}")
    print(f"Erros:            {total - acertos}")
    print(f"Acurácia:         {acuracia:.2%}")

    print("=" * 60)
    print("FIM DO TESTE")
    print("=" * 60)