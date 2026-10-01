import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from statsmodels.stats.proportion import proportion_confint


# ============================================================
# CONFIGURAÇÃO
# ============================================================

LIMIAR_ANALISE = 0.80

REFERENCIA_IC = {
    "retencao": {
        "count": 19_744,
        "nobs": 99_487,
    },
    "acima_threshold": {
        "count": 1_683,
        "nobs": 3_949,
    },
    "erro_intervencao": {

        "count": 222,
        "nobs": 1_683,

    },
}

ALPHA_IC = 0.10 


def _status_ic(valor: float, ic_inf: float, ic_sup: float) -> str:
    if valor < ic_inf:
        return "abaixo da referência"
    if valor > ic_sup:
        return "acima da referência"
    return "dentro da referência"


# ============================================================
# 0. RESUMO E SAÚDE DO PIPELINE (COM IC)
# ============================================================

def calcular_resumo(
    df_raw: pd.DataFrame,
    df_prev: pd.DataFrame,
    df_bot: pd.DataFrame,
) -> dict:
    """
    Calcula os indicadores gerais de volume e compara a taxa de
    retenção do pré-processamento e a taxa de mensagens acima do
    threshold com os intervalos de confiança de referência.
    """

    taxa_retencao = (
        len(df_prev) / len(df_raw)
        if len(df_raw) > 0
        else 0.0
    )

    df_acima_threshold = df_prev[
        df_prev["probabilidade"] >= LIMIAR_ANALISE
    ]

    taxa_acima_threshold = (
        len(df_acima_threshold) / len(df_prev)
        if len(df_prev) > 0
        else 0.0
    )

    taxa_intervencao = (
        len(df_bot) / len(df_prev)
        if len(df_prev) > 0
        else 0.0
    )

    ic_retencao_inf, ic_retencao_sup = proportion_confint(
        count=REFERENCIA_IC["retencao"]["count"],
        nobs=REFERENCIA_IC["retencao"]["nobs"],
        alpha=ALPHA_IC,
        method="wilson",
    )

    ic_threshold_inf, ic_threshold_sup = proportion_confint(
        count=REFERENCIA_IC["acima_threshold"]["count"],
        nobs=REFERENCIA_IC["acima_threshold"]["nobs"],
        alpha=ALPHA_IC,
        method="wilson",
    )

    ic_erro_intervencao_inf, ic_erro_intervencao_sup = proportion_confint(
        count=REFERENCIA_IC["erro_intervencao"]["count"],
        nobs=REFERENCIA_IC["erro_intervencao"]["nobs"],
        alpha=ALPHA_IC,
        method="wilson",
    )

    return {
        "raw": int(len(df_raw)),
        "previsoes": int(len(df_prev)),
        "mensagens_bot": int(len(df_bot)),

        "taxa_retencao": float(taxa_retencao),
        "taxa_acima_threshold": float(taxa_acima_threshold),
        "taxa_intervencao": float(taxa_intervencao),

        "ic_retencao": {
            "inf": float(ic_retencao_inf),
            "sup": float(ic_retencao_sup),
            "status": _status_ic(
                taxa_retencao,
                ic_retencao_inf,
                ic_retencao_sup,
            ),
        },

        "ic_acima_threshold": {
            "inf": float(ic_threshold_inf),
            "sup": float(ic_threshold_sup),
            "status": _status_ic(
                taxa_acima_threshold,
                ic_threshold_inf,
                ic_threshold_sup,
            ),
        },

        "ic_erro_intervencao": {
            "inf": float(ic_erro_intervencao_inf),
            "sup": float(ic_erro_intervencao_sup),
            "status": _status_ic(
                ic_erro_intervencao_inf,
                ic_erro_intervencao_sup,
                taxa_intervencao,
            ),
        },
    }


# ============================================================
# 1. SLAs DE LATÊNCIA
# ============================================================

def calcular_slas(
    df_raw: pd.DataFrame,
    df_prev: pd.DataFrame,
    df_bot: pd.DataFrame,
) -> dict:
    """
    Calcula as métricas de latência de cada etapa do pipeline de
    produção (ingestão -> predição -> geração LLM -> envio).
    """

    df_raw = df_raw.copy()
    df_prev = df_prev.copy()
    df_bot = df_bot.copy()

    df_raw["data_hora"] = pd.to_datetime(df_raw["data_hora"], utc=True)
    df_prev["processado_em"] = pd.to_datetime(df_prev["processado_em"], utc=True)
    df_bot["gerado_em"] = pd.to_datetime(df_bot["gerado_em"], utc=True)
    df_bot["enviado_em"] = pd.to_datetime(df_bot["enviado_em"], utc=True)

    # RAW -> PREVISÃO
    df_worker = df_prev.merge(
        df_raw[["id", "data_hora"]],
        left_on="mensagem_id",
        right_on="id",
        how="left",
        suffixes=("", "_raw"),
    )
    df_worker["latencia_worker_seg"] = (
        df_worker["processado_em"] - df_worker["data_hora"]
    ).dt.total_seconds()

    # PREVISÃO -> LLM
    df_bot_sla = df_bot.merge(
        df_prev[["id", "processado_em"]],
        left_on="previsao_id",
        right_on="id",
        how="left",
        suffixes=("", "_prev"),
    )
    df_bot_sla["latencia_llm_seg"] = (
        df_bot_sla["gerado_em"] - df_bot_sla["processado_em"]
    ).dt.total_seconds()

    # LLM -> DELIVERY
    df_bot_sla["latencia_envio_seg"] = (
        df_bot_sla["enviado_em"] - df_bot_sla["gerado_em"]
    ).dt.total_seconds()

    def resumir_latencia(serie):
        serie = serie.dropna()
        if serie.empty:
            return None
        return {
            "p50": float(serie.median()),
            "p95": float(serie.quantile(0.95)),
            "p99": float(serie.quantile(0.99)),
            "max": float(serie.max()),
            "n": int(len(serie)),
        }

    df_full_pipeline = pd.merge(
        df_worker,
        df_bot_sla,
        left_on="id",
        right_on="previsao_id",
        how="inner",
    )
    df_full_pipeline = df_full_pipeline.dropna(
        subset=["latencia_worker_seg", "latencia_llm_seg"]
    )

    if not df_full_pipeline.empty:
        df_full_pipeline["latencia_total_seg"] = (
            df_full_pipeline["latencia_worker_seg"]
            + df_full_pipeline["latencia_llm_seg"]
        )

    return {
        "worker": resumir_latencia(df_worker["latencia_worker_seg"]),
        "llm": resumir_latencia(df_bot_sla["latencia_llm_seg"]),
        "delivery": resumir_latencia(df_bot_sla["latencia_envio_seg"]),
        "total": resumir_latencia(
            df_full_pipeline["latencia_total_seg"]
            if not df_full_pipeline.empty
            else pd.Series(dtype=float)
        ),
        "df_worker": df_worker,
        "df_bot_sla": df_bot_sla,
        "df_full_pipeline": df_full_pipeline,
    }


# ============================================================
# 2. DESEMPENHO DO MODELO
# ============================================================

def calcular_desempenho(df_prev: pd.DataFrame) -> dict:
    if df_prev.empty:
        return {
            "macro_f1": None,
            "weighted_f1": None,
            "acuracia": None,
            "classification_report": {},
            "matriz_confusao": pd.DataFrame(),
        }

    y_real = df_prev["autor_real"]
    y_pred = df_prev["autor_predito"]

    labels = sorted(set(y_real).union(set(y_pred)))

    macro_f1 = f1_score(y_real, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_real, y_pred, average="weighted", zero_division=0)
    acuracia = float((y_real == y_pred).mean())

    relatorio = classification_report(
        y_real, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    matriz = confusion_matrix(y_real, y_pred, labels=labels)
    matriz = pd.DataFrame(matriz, index=labels, columns=labels)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "acuracia": acuracia,
        "classification_report": relatorio,
        "matriz_confusao": matriz,
    }


# ============================================================
# 3. CALIBRAÇÃO (CONFIANÇA x ACERTO)
# ============================================================

def calcular_calibracao(df_prev: pd.DataFrame) -> dict:
    df = df_prev.copy()

    if df.empty:
        return {}

    df["is_acerto"] = df["autor_real"] == df["autor_predito"]

    acertos = df.loc[df["is_acerto"], "probabilidade"]
    erros = df.loc[~df["is_acerto"], "probabilidade"]

    resultado = {
        "mediana_acertos": float(acertos.median()),
        "mediana_erros": float(erros.median()),
        "n_acertos": int(len(acertos)),
        "n_erros": int(len(erros)),
    }

    if len(acertos) > 0 and len(erros) > 0:
        u, p_value = stats.mannwhitneyu(acertos, erros, alternative="greater")

        n1, n2 = len(acertos), len(erros)
        rank_biserial = 1 - (2 * u) / (n1 * n2)
        abs_r = abs(rank_biserial)

        if abs_r < 0.14:
            efeito = "Insignificante"
        elif abs_r < 0.33:
            efeito = "Pequeno"
        elif abs_r < 0.47:
            efeito = "Médio"
        else:
            efeito = "Grande / Forte"

        resultado.update({
            "mann_whitney_u": float(u),
            "mann_whitney_p": float(p_value),
            "rank_biserial": float(rank_biserial),
            "efeito_rank_biserial": efeito,
            "significativo": bool(p_value < 0.05),
        })

    # Subconjunto de alta confiança (regra de negócio: P >= LIMIAR_ANALISE)
    df_alta_confianca = df[df["probabilidade"] >= LIMIAR_ANALISE].copy()

    if not df_alta_confianca.empty:
        resultado["alta_confianca"] = {
            "limiar": LIMIAR_ANALISE,
            "n": int(len(df_alta_confianca)),
            "cobertura": float(len(df_alta_confianca) / len(df)),
            "acuracia": float(
                (
                    df_alta_confianca["autor_real"]
                    == df_alta_confianca["autor_predito"]
                ).mean()
            ),
            "macro_f1": float(
                f1_score(
                    df_alta_confianca["autor_real"],
                    df_alta_confianca["autor_predito"],
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    df_alta_confianca["autor_real"],
                    df_alta_confianca["autor_predito"],
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    return resultado


# ============================================================
# 4. IMPOSTORES E DOPPELGÄNGERS
# ============================================================

def calcular_impostura(df_prev: pd.DataFrame) -> dict:
    df = df_prev.copy()

    if df.empty:
        return {}

    df["is_impostor"] = df["autor_real"] != df["autor_predito"]
    casos = df[df["is_impostor"]].copy()

    total = len(df)
    total_impostores = len(casos)

    top_falsificadores = (
        casos["autor_real"]
        .value_counts()
        .rename_axis("Membro")
        .reset_index(name="Vezes_Impostor")
    )

    top_clonados = (
        casos["autor_predito"]
        .value_counts()
        .rename_axis("Membro")
        .reset_index(name="Vezes_Clonado")
    )

    labels = sorted(set(df["autor_real"]).union(set(df["autor_predito"])))

    matriz_doppel = pd.crosstab(
        casos["autor_real"], casos["autor_predito"]
    ).reindex(index=labels, columns=labels, fill_value=0)

    return {
        "total": int(total),
        "impostores": int(total_impostores),
        "legitimos": int(total - total_impostores),
        "taxa_impostura": (
            float(total_impostores / total) if total > 0 else 0.0
        ),
        "top_falsificadores": top_falsificadores,
        "top_clonados": top_clonados,
        "matriz_doppel": matriz_doppel,
    }


# ============================================================
# 5. DISTRIBUIÇÃO DE CATEGORIAS
# ============================================================

def calcular_categorias(df_prev: pd.DataFrame) -> dict:
    df = df_prev.copy()

    if df.empty:
        return {"distribuicao": pd.DataFrame()}

    rotulo_sem_intervencao = f"SEM_INTERVENCAO (<{LIMIAR_ANALISE:.2f})"
    df["categoria_geral"] = df["categoria"].fillna(rotulo_sem_intervencao)

    distribuicao = (
        df["categoria_geral"]
        .value_counts()
        .rename_axis("Categoria")
        .reset_index(name="Quantidade")
    )
    distribuicao["Percentual"] = distribuicao["Quantidade"] / len(df) * 100

    return {"distribuicao": distribuicao}


# ============================================================
# 6. LLM E DELIVERY
# ============================================================

def calcular_llm(df_bot: pd.DataFrame) -> dict:
    df = df_bot.copy()

    if df.empty:
        return {
            "total_gerado": 0,
            "media_palavras": None,
            "media_caracteres": None,
            "max_caracteres": None,
            "max_palavras": None,
            "min_caracteres": None,
            "min_palavras": None,
            "personalidades": pd.DataFrame(),
            "status": {},
            "taxa_envio": 0.0,
            "textos_nulos": 0,
            "textos_vazios": 0,
            "df": df,
        }

    df["num_caracteres"] = df["texto"].fillna("").str.len()
    df["num_palavras"] = df["texto"].apply(
        lambda x: len(str(x).split()) if pd.notnull(x) else 0
    )

    stats_personalidade = (
        df.groupby("personalidade")
        .agg(
            total_gerado=("id", "count"),
            media_palavras=("num_palavras", "mean"),
            std_palavras=("num_palavras", "std"),
            min_palavras=("num_palavras", "min"),
            max_palavras=("num_palavras", "max"),
        )
        .reset_index()
    )
    stats_personalidade["std_palavras"] = stats_personalidade["std_palavras"].fillna(0)

    total_gerado = len(df)
    status_counts = df["status"].value_counts().to_dict()
    enviados = status_counts.get("ENVIADO", 0)
    taxa_envio = enviados / total_gerado if total_gerado > 0 else 0.0

    return {
        "total_gerado": int(total_gerado),
        "media_palavras": float(df["num_palavras"].mean()),
        "media_caracteres": float(df["num_caracteres"].mean()),
        "max_caracteres": int(df["num_caracteres"].max()),
        "max_palavras": int(df["num_palavras"].max()),
        "min_caracteres": int(df["num_caracteres"].min()),
        "min_palavras": int(df["num_palavras"].min()),
        "personalidades": stats_personalidade,
        "status": status_counts,
        "taxa_envio": float(taxa_envio),
        "textos_nulos": int(df["texto"].isnull().sum()),
        "textos_vazios": int((df["num_caracteres"] == 0).sum()),
        "df": df,
    }


# ============================================================
# 7. DRIFT TEMPORAL
# ============================================================

def calcular_drift(df_prev: pd.DataFrame) -> dict:
    df = df_prev.copy()

    if df.empty:
        return {
            "df_semanal": pd.DataFrame(),
            "df_completo": pd.DataFrame(),
            "qtd_semanas": 0,
        }

    df["processado_em"] = pd.to_datetime(df["processado_em"], utc=True)
    df["semana"] = df["processado_em"].dt.to_period("W").dt.to_timestamp()
    df["comprimento_char"] = df["mensagem"].fillna("").str.len()
    df["is_acerto"] = df["autor_real"] == df["autor_predito"]

    semanas = sorted(df["semana"].unique())

    f1_semanal = []
    for semana in semanas:
        sub = df[df["semana"] == semana]
        f1_semanal.append({
            "semana": semana.strftime("%Y-%m-%d"),
            "macro_f1": f1_score(
                sub["autor_real"], sub["autor_predito"],
                average="macro", zero_division=0,
            ),
            "qtd_mensagens": len(sub),
        })

    df_semanal = pd.DataFrame(f1_semanal)

    resultado = {
        "df_semanal": df_semanal,
        "df_completo": df,
        "qtd_semanas": len(semanas),
        "semanas": semanas,
    }

    if len(semanas) >= 2:
        semana_anterior = semanas[-2]
        semana_atual = semanas[-1]

        anterior = df[df["semana"] == semana_anterior]
        atual = df[df["semana"] == semana_atual]

        ks_stat, ks_pvalue = stats.ks_2samp(
            anterior["comprimento_char"].dropna(),
            atual["comprimento_char"].dropna(),
        )

        tabela = pd.crosstab(df["semana"], df["autor_real"])
        chi2_stat, chi2_pvalue, _, _ = stats.chi2_contingency(tabela)

        resultado["comparacao_ultimas_semanas"] = {
            "semana_anterior": semana_anterior,
            "semana_atual": semana_atual,
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "chi2_stat": float(chi2_stat),
            "chi2_pvalue": float(chi2_pvalue),
        }

    return resultado


# ============================================================
# ORQUESTRADOR
# ============================================================

def calcular_metricas(
    df_raw: pd.DataFrame,
    df_prev: pd.DataFrame,
    df_bot: pd.DataFrame,
) -> dict:
    """
    Executa todas as métricas utilizadas no relatório semanal.
    """

    return {
        "resumo": calcular_resumo(df_raw, df_prev, df_bot),
        "slas": calcular_slas(df_raw, df_prev, df_bot),
        "desempenho": calcular_desempenho(df_prev),
        "calibracao": calcular_calibracao(df_prev),
        "impostura": calcular_impostura(df_prev),
        "categorias": calcular_categorias(df_prev),
        "llm": calcular_llm(df_bot),
        "drift": calcular_drift(df_prev),
    }