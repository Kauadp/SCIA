import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="notebook")


# ============================================================
# 1. DESEMPENHO
# ============================================================

def grafico_matriz_confusao(matriz: pd.DataFrame):
    if matriz.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Autor Predito")
    ax.set_ylabel("Autor Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def grafico_f1_por_membro(classification_report: dict):
    dados = []
    for membro, metricas in classification_report.items():
        if membro in ["accuracy", "macro avg", "weighted avg"]:
            continue
        dados.append({"Membro": membro, "F1": metricas["f1-score"]})

    if not dados:
        return None

    df = pd.DataFrame(dados).sort_values("F1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df, x="F1", y="Membro", ax=ax)
    ax.set_title("F1-score por membro")
    ax.set_xlabel("F1-score")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


# ============================================================
# 2. CONFIANÇA E SELETIVIDADE
# ============================================================

def grafico_confianca_acerto(df_prev: pd.DataFrame):
    if df_prev.empty:
        return None

    df = df_prev.copy()
    df["Resultado"] = (df["autor_real"] == df["autor_predito"]).map(
        {True: "Acerto", False: "Erro"}
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df, x="probabilidade", hue="Resultado", bins=20,
        stat="density", common_norm=False, element="step", fill=False,
        ax=ax,
    )
    ax.set_title("Distribuição de confiança por resultado")
    ax.set_xlabel("Confiança da previsão")
    ax.set_ylabel("Densidade")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def grafico_acuracia_por_faixa(df_prev: pd.DataFrame):
    if df_prev.empty:
        return None

    df = df_prev.copy()

    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [
        "0–30%", "30–40%", "40–50%", "50–60%",
        "60–70%", "70–80%", "80–90%", "90–100%",
    ]

    df["faixa_confianca"] = pd.cut(
        df["probabilidade"], bins=bins, labels=labels, include_lowest=True
    )
    df["acerto"] = df["autor_real"] == df["autor_predito"]

    resultado = (
        df.groupby("faixa_confianca", observed=False)
        .agg(acuracia=("acerto", "mean"), quantidade=("acerto", "size"))
        .reset_index()
    )
    resultado["acuracia"] *= 100

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=resultado, x="faixa_confianca", y="acuracia", ax=ax)
    ax.set_title("Acurácia por faixa de confiança")
    ax.set_xlabel("Faixa de confiança")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(0, 100)

    for i, row in resultado.iterrows():
        ax.text(
            i, row["acuracia"] + 2,
            f'{row["acuracia"]:.1f}%\nn={row["quantidade"]}',
            ha="center", va="bottom",
        )

    fig.tight_layout()
    return fig


def grafico_cobertura_precisao(df_prev: pd.DataFrame):
    if df_prev.empty:
        return None

    df = df_prev.copy()
    limiares = [0.50, 0.60, 0.70, 0.80, 0.90]
    resultados = []

    for limiar in limiares:
        selecionados = df[df["probabilidade"] >= limiar]
        if selecionados.empty:
            continue

        precisao = (
            selecionados["autor_real"] == selecionados["autor_predito"]
        ).mean()
        cobertura = len(selecionados) / len(df)

        resultados.append({
            "limiar": limiar,
            "precisao": precisao * 100,
            "cobertura": cobertura * 100,
        })

    resultado = pd.DataFrame(resultados)
    if resultado.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(resultado["cobertura"], resultado["precisao"], marker="o")

    for _, row in resultado.iterrows():
        ax.annotate(
            f'{row["limiar"]:.0%}',
            (row["cobertura"], row["precisao"]),
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_title("Precisão × cobertura por limiar")
    ax.set_xlabel("Cobertura (%)")
    ax.set_ylabel("Precisão (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


# ============================================================
# 3. IMPOSTORES E DOPPELGÄNGERS
# ============================================================

def grafico_impostores(top_falsificadores: pd.DataFrame):
    if top_falsificadores.empty:
        return None

    df = top_falsificadores.sort_values("Vezes_Impostor", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df, x="Vezes_Impostor", y="Membro", ax=ax)
    ax.set_title("Ocorrências de impostura por membro")
    ax.set_xlabel("Quantidade de ocorrências")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def grafico_clonados(top_clonados: pd.DataFrame):
    if top_clonados.empty:
        return None

    df = top_clonados.sort_values("Vezes_Clonado", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df, x="Vezes_Clonado", y="Membro", ax=ax)
    ax.set_title("Identidades mais frequentemente imitadas")
    ax.set_xlabel("Quantidade de ocorrências")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def grafico_doppelganger(matriz_doppel: pd.DataFrame):
    """Mapa de calor de quem é confundido com quem (apenas erros)."""
    if matriz_doppel.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        matriz_doppel, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="lightgrey", ax=ax,
        cbar_kws={"label": "Frequência de confusão"},
    )
    ax.set_title("Matriz de Doppelgängers (quem é confundido com quem)")
    ax.set_xlabel("Estilo atribuído (autor predito)")
    ax.set_ylabel("Autor real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def grafico_distribuicao_categorias(distribuicao: pd.DataFrame):
    """Distribuição das categorias de previsão (SEM_INTERVENCAO, ERRO_*, ACERTO_*)."""
    if distribuicao.empty:
        return None

    df = distribuicao.sort_values("Quantidade", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Quantidade", y="Categoria", ax=ax)
    ax.set_title("Distribuição de categorias das previsões")
    ax.set_xlabel("Quantidade de previsões")
    ax.set_ylabel("")

    for i, row in df.iterrows():
        ax.text(
            row["Quantidade"] + 0.5, i,
            f'{row["Quantidade"]} ({row["Percentual"]:.1f}%)',
            va="center",
        )

    fig.tight_layout()
    return fig


# ============================================================
# 4. LLM E DELIVERY
# ============================================================

def grafico_boxplot_personalidade(df_bot: pd.DataFrame):
    """Variabilidade de tamanho de resposta por personalidade."""
    if df_bot.empty or "num_palavras" not in df_bot.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_bot, x="personalidade", y="num_palavras", ax=ax)
    ax.set_title("Variação de tamanho de resposta por personalidade")
    ax.set_xlabel("")
    ax.set_ylabel("Quantidade de palavras")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def grafico_latencias_llm_delivery(df_bot_sla: pd.DataFrame):
    """Densidade das latências LLM (predição->geração) e Delivery (geração->envio)."""
    if df_bot_sla.empty:
        return None

    tem_llm = (
        "latencia_llm_seg" in df_bot_sla.columns
        and df_bot_sla["latencia_llm_seg"].notna().any()
    )
    tem_envio = (
        "latencia_envio_seg" in df_bot_sla.columns
        and df_bot_sla["latencia_envio_seg"].notna().any()
    )

    if not tem_llm and not tem_envio:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if tem_llm:
        sns.kdeplot(
            data=df_bot_sla["latencia_llm_seg"].dropna(),
            label="Latência LLM (predição → geração)",
            fill=True, color="#2ca02c", ax=ax,
        )

    if tem_envio:
        sns.kdeplot(
            data=df_bot_sla["latencia_envio_seg"].dropna(),
            label="Latência Delivery (geração → envio)",
            fill=True, color="#d62728", ax=ax,
        )

    ax.set_title("Distribuição de latências (LLM vs Delivery)")
    ax.set_xlabel("Segundos")
    ax.set_ylabel("Densidade")
    ax.legend()
    fig.tight_layout()
    return fig


# ============================================================
# 5. DRIFT TEMPORAL
# ============================================================

def grafico_f1_semanal(df_semanal: pd.DataFrame):
    if df_semanal.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=df_semanal, x="semana", y="macro_f1", marker="o", ax=ax)
    ax.set_title("Evolução semanal do Macro F1")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


def grafico_drift_comprimento(df_completo: pd.DataFrame, qtd_semanas: int):
    """Distribuição do comprimento de mensagens, comparando semanas quando houver mais de uma."""
    if df_completo.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if qtd_semanas >= 2:
        sns.kdeplot(
            data=df_completo, x="comprimento_char", hue="semana",
            common_norm=False, fill=True, palette="Set1", ax=ax,
        )
        ax.set_title("Distribuição do comprimento de mensagens\n(comparação semanal)")
    else:
        sns.histplot(
            data=df_completo, x="comprimento_char", kde=True,
            color="#2b5c8f", ax=ax,
        )
        ax.set_title("Distribuição do comprimento de mensagens\n(semana baseline)")

    ax.set_xlabel("Número de caracteres por mensagem")
    ax.set_ylabel("Densidade")
    fig.tight_layout()
    return fig


def grafico_drift_resultado(df_completo: pd.DataFrame):
    """Volume semanal de previsões, separado por acerto/erro."""
    if df_completo.empty or "semana" not in df_completo.columns:
        return None

    df = df_completo.copy()
    df["Resultado"] = df["is_acerto"].map({True: "Acerto", False: "Erro"})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df, x="semana", hue="Resultado",
        palette={"Acerto": "#2ca02c", "Erro": "#d62728"}, ax=ax,
    )
    ax.set_title("Volume semanal de previsões por resultado")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Quantidade")
    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


# ============================================================
# ORQUESTRADOR
# ============================================================

def gerar_graficos(metricas: dict, df_prev: pd.DataFrame) -> dict:
    """
    Gera todos os gráficos utilizados no relatório semanal.
    """

    desempenho = metricas["desempenho"]
    impostura = metricas["impostura"]
    categorias = metricas["categorias"]
    llm = metricas["llm"]
    slas = metricas["slas"]
    drift = metricas["drift"]

    return {
        # 1. Desempenho
        "matriz_confusao": grafico_matriz_confusao(
            desempenho["matriz_confusao"]
        ),
        "f1_por_membro": grafico_f1_por_membro(
            desempenho["classification_report"]
        ),

        # 2. Confiança
        "confianca_acerto": grafico_confianca_acerto(df_prev),
        "acuracia_por_faixa": grafico_acuracia_por_faixa(df_prev),
        "cobertura_precisao": grafico_cobertura_precisao(df_prev),

        # 3. Impostores
        "impostores": grafico_impostores(impostura["top_falsificadores"]),
        "clonados": grafico_clonados(impostura["top_clonados"]),
        "doppelganger": grafico_doppelganger(impostura["matriz_doppel"]),
        "distribuicao_categorias": grafico_distribuicao_categorias(
            categorias["distribuicao"]
        ),

        # 4. LLM e delivery
        "boxplot_personalidade": grafico_boxplot_personalidade(llm["df"]),
        "latencias_llm_delivery": grafico_latencias_llm_delivery(
            slas["df_bot_sla"]
        ),

        # 5. Drift
        "f1_semanal": grafico_f1_semanal(drift["df_semanal"]),
        "drift_comprimento": grafico_drift_comprimento(
            drift["df_completo"], drift["qtd_semanas"]
        ),
        "drift_resultado": grafico_drift_resultado(drift["df_completo"]),
    }