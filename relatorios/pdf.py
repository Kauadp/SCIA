from pathlib import Path

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PASTA_RELATORIOS = BASE_DIR / "relatorios"
PASTA_GRAFICOS = PASTA_RELATORIOS / "graficos"

PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS DE FORMATAÇÃO
# ============================================================

def formatar_percentual(valor):
    if valor is None:
        return "-"
    return f"{valor * 100:.1f}%"


def formatar_segundos(valor):
    if valor is None:
        return "-"
    if valor < 60:
        return f"{valor:.2f}s"
    return f"{valor / 60:.2f} min"


def formatar_pvalor(valor):
    if valor is None:
        return "-"
    return f"{valor:.3e}"


def formatar_numero(valor, casas=1, sufixo=""):
    if valor is None:
        return "-"
    return f"{valor:.{casas}f}{sufixo}"


def salvar_grafico(fig, nome):
    if fig is None:
        return None

    caminho = PASTA_GRAFICOS / f"{nome}.png"
    fig.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return caminho


def adicionar_titulo(story, texto, styles):
    story.append(Paragraph(texto, styles["TituloSecao"]))
    story.append(Spacer(1, 0.25 * cm))


def adicionar_texto(story, texto, styles):
    story.append(Paragraph(texto, styles["TextoRelatorio"]))


def adicionar_grafico(story, fig, nome, largura=17 * cm):
    caminho = salvar_grafico(fig, nome)
    if caminho is None:
        return

    imagem = Image(str(caminho))
    proporcao = imagem.imageHeight / imagem.imageWidth
    imagem.drawWidth = largura
    imagem.drawHeight = largura * proporcao

    story.append(imagem)
    story.append(Spacer(1, 0.4 * cm))


def tabela_simples(dados, larguras=None):
    tabela = Table(dados, colWidths=larguras, repeatRows=1)
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tabela


def linha_latencia(nome, dados):
    if dados is None:
        return [nome, "-", "-", "-", "-", "-"]
    return [
        nome,
        formatar_segundos(dados["p50"]),
        formatar_segundos(dados["p95"]),
        formatar_segundos(dados["p99"]),
        formatar_segundos(dados["max"]),
        str(dados["n"]),
    ]


# ============================================================
# CONSTRUÇÃO DO PDF
# ============================================================

def construir_pdf(metricas, graficos, caminho_saida):
    """
    Constrói o relatório semanal em PDF.
    """

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="TituloRelatorio", parent=styles["Title"],
        alignment=TA_CENTER, fontSize=20, leading=24, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Subtitulo", parent=styles["Normal"],
        alignment=TA_CENTER, fontSize=10, textColor=colors.grey, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="TituloSecao", parent=styles["Heading2"],
        fontSize=14, leading=18, spaceBefore=12, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="TextoRelatorio", parent=styles["BodyText"],
        fontSize=9, leading=13, spaceAfter=6,
    ))

    doc = SimpleDocTemplate(
        str(caminho_saida), pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )

    story = []

    # ============================================================
    # CAPA
    # ============================================================

    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("SCIA", styles["TituloRelatorio"]))
    story.append(Paragraph("Relatório Semanal de Produção", styles["TituloRelatorio"]))
    story.append(Paragraph(
        "Análise operacional e desempenho do sistema de detecção",
        styles["Subtitulo"],
    ))

    resumo = metricas["resumo"]

    dados_resumo = [
        ["Mensagens", "Previsões", "Intervenções"],
        [str(resumo["raw"]), str(resumo["previsoes"]), str(resumo["mensagens_bot"])],
    ]
    story.append(tabela_simples(dados_resumo, larguras=[5.5 * cm] * 3))
    story.append(Spacer(1, 1 * cm))

    # ============================================================
    # 0. SAÚDE DO PIPELINE
    # ============================================================

    adicionar_titulo(story, "0. Resumo e saúde do pipeline", styles)

    dados_taxas = [
        ["Indicador", "Valor", "Referência (IC 90%)", "Status"],
        [
            "Retenção do pré-processamento",
            formatar_percentual(resumo["taxa_retencao"]),
            f'{resumo["ic_retencao"]["inf"] * 100:.2f}% – '
            f'{resumo["ic_retencao"]["sup"] * 100:.2f}%',
            resumo["ic_retencao"]["status"],
        ],
        [
            f"Mensagens acima do threshold",
            formatar_percentual(resumo["taxa_acima_threshold"]),
            f'{resumo["ic_acima_threshold"]["inf"] * 100:.2f}% – '
            f'{resumo["ic_acima_threshold"]["sup"] * 100:.2f}%',
            resumo["ic_acima_threshold"]["status"],
        ],
        [
            "Taxa de intervenção",
            formatar_percentual(resumo["taxa_intervencao"]),
            f'{resumo["ic_erro_intervencao"]["inf"] * 100:.2f}% – '
            f'{resumo["ic_erro_intervencao"]["sup"] * 100:.2f}%',
            resumo["ic_erro_intervencao"]["status"],
        ],
    ]

    story.append(tabela_simples(dados_taxas, larguras=[6 * cm, 3 * cm, 4.5 * cm, 3.5 * cm]))
    story.append(Spacer(1, 0.5 * cm))

    slas = metricas["slas"]

    dados_sla = [
        ["Etapa", "p50", "p95", "p99", "Máximo", "N"],
        linha_latencia("Worker ML (ingestão → predição)", slas["worker"]),
        linha_latencia("LLM (predição → geração)", slas["llm"]),
        linha_latencia("Delivery (geração → envio)", slas["delivery"]),
        linha_latencia("Pipeline total", slas["total"]),
    ]

    story.append(tabela_simples(
        dados_sla, larguras=[6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]
    ))

    adicionar_texto(
        story,
        "Nota: valores de p95/p99/máximo muito acima da mediana costumam indicar "
        "eventos pontuais (reinícios, backlog acumulado) e não necessariamente o "
        "comportamento típico do pipeline. Investigar separadamente antes de tratar "
        "como regressão.",
        styles,
    )

    # ============================================================
    # 1. DESEMPENHO
    # ============================================================

    story.append(PageBreak())
    adicionar_titulo(story, "1. Desempenho do modelo", styles)

    desempenho = metricas["desempenho"]

    dados_desempenho = [
        ["Métrica", "Valor"],
        ["Acurácia", formatar_percentual(desempenho["acuracia"])],
        ["Macro F1", formatar_percentual(desempenho["macro_f1"])],
        ["Weighted F1", formatar_percentual(desempenho["weighted_f1"])],
    ]
    story.append(tabela_simples(dados_desempenho, larguras=[8 * cm, 8 * cm]))
    story.append(Spacer(1, 0.5 * cm))

    adicionar_grafico(story, graficos["matriz_confusao"], "matriz_confusao")
    adicionar_grafico(story, graficos["f1_por_membro"], "f1_por_membro")

    # ============================================================
    # 2. CONFIANÇA E SELETIVIDADE
    # ============================================================

    story.append(PageBreak())
    adicionar_titulo(story, "2. Confiança e seletividade", styles)

    calibracao = metricas["calibracao"]

    dados_confianca = [
        ["Métrica", "Valor"],
        ["Mediana dos acertos", f'{calibracao["mediana_acertos"]:.2%}'],
        ["Mediana dos erros", f'{calibracao["mediana_erros"]:.2%}'],
        ["Mann-Whitney p", formatar_pvalor(calibracao.get("mann_whitney_p"))],
        [
            "Diferença estatisticamente significativa?",
            "Sim" if calibracao.get("significativo") else "Não",
        ],
        [
            "Tamanho do efeito (rank-biserial)",
            calibracao.get("efeito_rank_biserial", "-"),
        ],
    ]
    story.append(tabela_simples(dados_confianca, larguras=[9 * cm, 7 * cm]))

    alta = calibracao.get("alta_confianca")
    if alta:
        story.append(Spacer(1, 0.4 * cm))
        dados_alta = [
            ["Métrica", f'P ≥ {alta["limiar"]:.2f}'],
            ["Cobertura", formatar_percentual(alta["cobertura"])],
            ["Acurácia", formatar_percentual(alta["acuracia"])],
            ["Macro F1", formatar_percentual(alta["macro_f1"])],
            ["Weighted F1", formatar_percentual(alta["weighted_f1"])],
        ]
        story.append(tabela_simples(dados_alta, larguras=[8 * cm, 8 * cm]))

    adicionar_grafico(story, graficos["confianca_acerto"], "confianca_acerto")
    adicionar_grafico(story, graficos["acuracia_por_faixa"], "acuracia_por_faixa")
    adicionar_grafico(story, graficos["cobertura_precisao"], "cobertura_precisao")

    # ============================================================
    # 3. IMPOSTORES
    # ============================================================

    story.append(PageBreak())
    adicionar_titulo(story, "3. Impostores e identidades imitadas", styles)

    impostura = metricas["impostura"]

    dados_impostura = [
        ["Indicador", "Quantidade"],
        ["Previsões avaliadas", str(impostura["total"])],
        ["Ocorrências de impostura", str(impostura["impostores"])],
        ["Ocorrências legítimas", str(impostura["legitimos"])],
        ["Taxa de impostura", formatar_percentual(impostura["taxa_impostura"])],
    ]
    story.append(tabela_simples(dados_impostura, larguras=[8 * cm, 8 * cm]))

    adicionar_grafico(story, graficos["impostores"], "impostores")
    adicionar_grafico(story, graficos["clonados"], "clonados")
    adicionar_grafico(story, graficos["doppelganger"], "doppelganger")

    categorias = metricas["categorias"]["distribuicao"]
    if not categorias.empty:
        story.append(Spacer(1, 0.3 * cm))

        dados_categorias = [["Categoria", "Quantidade", "Percentual"]]
        for _, row in categorias.sort_values("Quantidade", ascending=False).iterrows():
            dados_categorias.append([
                row["Categoria"], str(int(row["Quantidade"])),
                f'{row["Percentual"]:.1f}%',
            ])

        story.append(tabela_simples(dados_categorias, larguras=[8 * cm, 4 * cm, 4 * cm]))

    adicionar_grafico(story, graficos["distribuicao_categorias"], "distribuicao_categorias")

    # ============================================================
    # 4. LLM E ENTREGA
    # ============================================================

    story.append(PageBreak())
    adicionar_titulo(story, "4. LLM e entrega das intervenções", styles)

    llm = metricas["llm"]

    maior_mensagem = (
        f'{llm["max_palavras"]} palavras'
        if llm["max_palavras"] is not None
        else "-"
    )

    dados_llm = [
        ["Métrica", "Valor"],
        ["Mensagens geradas", str(llm["total_gerado"])],
        ["Média de palavras", formatar_numero(llm["media_palavras"])],
        ["Média de caracteres", formatar_numero(llm["media_caracteres"])],
        ["Maior mensagem", maior_mensagem],
        ["Taxa de envio efetivo", formatar_percentual(llm["taxa_envio"])],
        ["Textos nulos", str(llm["textos_nulos"])],
        ["Textos vazios", str(llm["textos_vazios"])],
    ]
    story.append(tabela_simples(dados_llm, larguras=[8 * cm, 8 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    status = llm["status"]
    if status:
        dados_status = [["Status de envio", "Quantidade"]]
        for nome_status, quantidade in status.items():
            dados_status.append([nome_status, str(quantidade)])

        story.append(tabela_simples(dados_status, larguras=[8 * cm, 8 * cm]))
        story.append(Spacer(1, 0.5 * cm))

    personalidades = llm["personalidades"]
    if not personalidades.empty:
        dados_personalidades = [["Personalidade", "Mensagens", "Média palavras"]]
        for _, row in personalidades.iterrows():
            dados_personalidades.append([
                row["personalidade"],
                str(int(row["total_gerado"])),
                f'{row["media_palavras"]:.1f}',
            ])

        story.append(tabela_simples(
            dados_personalidades, larguras=[6 * cm, 5 * cm, 5 * cm]
        ))

    adicionar_grafico(story, graficos["boxplot_personalidade"], "boxplot_personalidade")
    adicionar_grafico(story, graficos["latencias_llm_delivery"], "latencias_llm_delivery")

    # ============================================================
    # 5. DRIFT TEMPORAL
    # ============================================================

    story.append(PageBreak())
    adicionar_titulo(story, "5. Drift temporal", styles)

    drift = metricas["drift"]

    if "comparacao_ultimas_semanas" in drift:
        comparacao = drift["comparacao_ultimas_semanas"]

        dados_drift = [
            ["Teste", "Estatística", "p-value"],
            [
                "Kolmogorov-Smirnov",
                f'{comparacao["ks_stat"]:.4f}',
                formatar_pvalor(comparacao["ks_pvalue"]),
            ],
            [
                "Qui-quadrado",
                f'{comparacao["chi2_stat"]:.4f}',
                formatar_pvalor(comparacao["chi2_pvalue"]),
            ],
        ]
        story.append(tabela_simples(dados_drift, larguras=[6 * cm, 5 * cm, 5 * cm]))
    else:
        adicionar_texto(
            story,
            "Apenas uma semana de dados disponível até o momento — esta janela "
            "foi registrada como baseline. A comparação estatística entre semanas "
            "fica habilitada a partir da segunda semana de produção.",
            styles,
        )

    adicionar_grafico(story, graficos["f1_semanal"], "f1_semanal")
    adicionar_grafico(story, graficos["drift_comprimento"], "drift_comprimento")
    adicionar_grafico(story, graficos["drift_resultado"], "drift_resultado")

    # ============================================================
    # FINAL
    # ============================================================

    doc.build(story)