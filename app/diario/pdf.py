from pathlib import Path

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CARTAZES_DIR = BASE_DIR / "app" / "cartazes" / "cartazes_atualizados"
RELATORIOS_DIR = BASE_DIR / "app" / "diario"

PDF_PATH = RELATORIOS_DIR / "diario_do_inspetor.pdf"


# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------

PAPEL = colors.HexColor("#E8E0CE")
PAPEL_ESCURO = colors.HexColor("#D6CBB3")
TINTA = colors.HexColor("#171614")
TINTA_SUAVE = colors.HexColor("#37332C")
VERMELHO = colors.HexColor("#7B1E1E")
VERMELHO_CLARO = colors.HexColor("#A52A2A")
CINZA = colors.HexColor("#6C675D")
BRANCO_SUJO = colors.HexColor("#F1EBDD")
PRETO = colors.HexColor("#0C0B0A")


# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------

def _registrar_fontes() -> dict[str, str]:
    """
    Tenta registrar fontes comuns do Linux.

    O PDF continua funcionando mesmo que alguma delas não exista.
    """
    fontes = {
        "serif": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        ],
        "serif_bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        ],
        "mono": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ],
        "mono_bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
        ],
    }

    registrados = {}

    for nome, candidatos in fontes.items():
        for caminho in candidatos:
            path = Path(caminho)
            if path.exists():
                alias = f"DI_{nome}"
                pdfmetrics.registerFont(TTFont(alias, str(path)))
                registrados[nome] = alias
                break

    registrados.setdefault("serif", "Times-Roman")
    registrados.setdefault("serif_bold", "Times-Bold")
    registrados.setdefault("mono", "Courier")
    registrados.setdefault("mono_bold", "Courier-Bold")

    return registrados


FONTES = _registrar_fontes()


# ---------------------------------------------------------------------------
# Utilidades visuais
# ---------------------------------------------------------------------------

def _paragrafo(
    c: canvas.Canvas,
    texto: str,
    x: float,
    y: float,
    largura: float,
    *,
    fonte: str,
    tamanho: float,
    entrelinha: float,
    cor=TINTA,
    alinhamento=TA_LEFT,
) -> float:
    """Desenha um Paragraph e retorna a nova coordenada Y."""
    estilo = ParagraphStyle(
        name="diario",
        fontName=fonte,
        fontSize=tamanho,
        leading=entrelinha,
        textColor=cor,
        alignment=alinhamento,
        splitLongWords=False,
    )

    p = Paragraph(texto, estilo)
    _, altura = p.wrap(largura, A4[1])
    p.drawOn(c, x, y - altura)

    return y - altura


def _linha(c, x1, y1, x2, y2, cor=TINTA, espessura=0.7):
    c.setStrokeColor(cor)
    c.setLineWidth(espessura)
    c.line(x1, y1, x2, y2)


def _papel(c):
    largura, altura = A4

    c.setFillColor(PAPEL)
    c.rect(0, 0, largura, altura, fill=1, stroke=0)

    # Borda externa de página impressa.
    c.setStrokeColor(TINTA_SUAVE)
    c.setLineWidth(0.8)
    c.rect(9*mm, 9*mm, largura - 18*mm, altura - 18*mm, fill=0, stroke=1)

    c.setStrokeColor(CINZA)
    c.setLineWidth(0.35)
    c.rect(12*mm, 12*mm, largura - 24*mm, altura - 24*mm, fill=0, stroke=1)


def _ruido_papel(c, seed=1):
    """
    Textura discreta construída com pequenos pontos.
    Não depende de imagens externas.
    """
    import random

    random.seed(seed)
    largura, altura = A4

    c.saveState()
    c.setFillColor(colors.Color(0.15, 0.13, 0.10, alpha=0.045))

    for _ in range(1800):
        x = random.uniform(12*mm, largura - 12*mm)
        y = random.uniform(12*mm, altura - 12*mm)
        r = random.choice([0.12, 0.16, 0.22])
        c.circle(x, y, r, fill=1, stroke=0)

    c.restoreState()


def _carimbo(c, texto, x, y, rotacao=-8):
    c.saveState()
    c.translate(x, y)
    c.rotate(rotacao)

    c.setStrokeColor(VERMELHO)
    c.setLineWidth(1.2)
    c.rect(-32*mm, -7*mm, 64*mm, 14*mm, fill=0, stroke=1)

    c.setFillColor(VERMELHO)
    c.setFont(FONTES["mono_bold"], 9)
    c.drawCentredString(0, -3, texto)

    c.restoreState()


def _cabecalho(c, pagina, subtitulo=None):
    largura, altura = A4

    c.setFillColor(TINTA)
    c.setFont(FONTES["mono_bold"], 7.5)
    c.drawString(18*mm, altura - 19*mm, "ARQUIVO SCIA  /  DIÁRIO DO INSPETOR")

    c.setFont(FONTES["mono"], 7)
    c.drawRightString(
        largura - 18*mm,
        altura - 19*mm,
        f"FOLHA {pagina:02d} / 03",
    )

    _linha(
        c,
        18*mm,
        altura - 22*mm,
        largura - 18*mm,
        altura - 22*mm,
        TINTA,
        0.6,
    )

    if subtitulo:
        c.setFillColor(CINZA)
        c.setFont(FONTES["mono"], 6.8)
        c.drawString(18*mm, altura - 27*mm, subtitulo.upper())


def _rodape(c, texto="DOCUMENTO INTERNO  •  REGISTRO DIÁRIO"):
    largura, _ = A4

    _linha(c, 18*mm, 17*mm, largura - 18*mm, 17*mm, CINZA, 0.4)

    c.setFillColor(CINZA)
    c.setFont(FONTES["mono"], 6.5)
    c.drawString(18*mm, 12.5*mm, texto)

    c.drawRightString(
        largura - 18*mm,
        12.5*mm,
        "SCIA",
    )


# ---------------------------------------------------------------------------
# Página 1
# ---------------------------------------------------------------------------

def _pagina_capa(c, data_inicio: datetime, data_fim: datetime):
    largura, altura = A4

    _papel(c)
    _ruido_papel(c, seed=17)

    # Bloco de identificação.
    c.setFillColor(TINTA)
    c.setFont(FONTES["mono_bold"], 8)
    c.drawString(22*mm, altura - 30*mm, "ARQUIVO Nº SCIA-DIARIO")

    c.setFont(FONTES["mono"], 7)
    c.drawRightString(
        largura - 22*mm,
        altura - 30*mm,
        "CLASSIFICAÇÃO: INTERNO",
    )

    # Linha vermelha de arquivo.
    _linha(
        c,
        22*mm,
        altura - 35*mm,
        largura - 22*mm,
        altura - 35*mm,
        VERMELHO,
        2.2,
    )

    # Grande título.
    c.setFillColor(TINTA)
    c.setFont(FONTES["serif_bold"], 34)
    c.drawCentredString(largura / 2, altura - 105*mm, "DIÁRIO")

    c.setFont(FONTES["serif_bold"], 43)
    c.drawCentredString(largura / 2, altura - 124*mm, "DO INSPETOR")

    _linha(
        c,
        56*mm,
        altura - 132*mm,
        largura - 56*mm,
        altura - 132*mm,
        TINTA,
        0.9,
    )

    c.setFillColor(CINZA)
    c.setFont(FONTES["mono"], 8)
    c.drawCentredString(
        largura / 2,
        altura - 143*mm,
        "REGISTRO CRONOLÓGICO DE OCORRÊNCIAS",
    )

    # Intervalo.
    inicio = data_inicio.strftime("%d.%m.%Y")
    fim = data_fim.strftime("%d.%m.%Y")

    c.setFillColor(TINTA)
    c.setFont(FONTES["serif"], 15)
    c.drawCentredString(
        largura / 2,
        altura - 178*mm,
        f"{inicio}  —  {fim}",
    )

    # Ficha de investigação.
    ficha_y = altura - 207*mm
    ficha_h = 42*mm

    c.setStrokeColor(TINTA)
    c.setLineWidth(0.8)
    c.rect(42*mm, ficha_y - ficha_h, largura - 84*mm, ficha_h, fill=0, stroke=1)

    c.setFillColor(TINTA_SUAVE)
    c.setFont(FONTES["mono_bold"], 7)
    c.drawString(49*mm, ficha_y - 9*mm, "NATUREZA DO REGISTRO")

    c.setFont(FONTES["serif"], 11)
    c.drawString(49*mm, ficha_y - 18*mm, "Identificação e acompanhamento de imposturas")

    c.setFont(FONTES["mono_bold"], 7)
    c.drawString(49*mm, ficha_y - 29*mm, "ORIGEM")

    c.setFont(FONTES["serif"], 11)
    c.drawString(49*mm, ficha_y - 38*mm, "Sistema de vigilância SCIA")

    _carimbo(
        c,
        "EM INVESTIGAÇÃO",
        largura - 57*mm,
        49*mm,
        rotacao=-7,
    )

    _rodape(c)
    c.showPage()


# ---------------------------------------------------------------------------
# Página 2
# ---------------------------------------------------------------------------

def _desenhar_cartaz_mini(
    c,
    membro: str,
    recompensa: int,
    x: float,
    y: float,
    largura: float,
    altura: float,
    imagem: Path | None,
    *,
    show_meta: bool = True,
):
    """Renderiza o cartaz: em página de evidence, apenas a imagem; no restante, mantém o layout antigo."""
    c.saveState()

    if show_meta:
        # Layout original quando usado em outras páginas.
        c.setFillColor(BRANCO_SUJO)
        c.setStrokeColor(TINTA_SUAVE)
        c.setLineWidth(0.8)
        c.rect(x, y, largura, altura, fill=1, stroke=1)

        c.setFillColor(TINTA)
        c.rect(x, y + altura - 9*mm, largura, 9*mm, fill=1, stroke=0)

        c.setFillColor(BRANCO_SUJO)
        c.setFont(FONTES["mono_bold"], 6.5)
        c.drawString(x + 3*mm, y + altura - 6.2*mm, "PROCURADO")

        margem = 3.5*mm
        foto_x = x + margem
        foto_y = y + 18*mm
        foto_h = altura - 31*mm
        foto_w = largura - 2*margem

        if imagem and imagem.exists():
            try:
                c.drawImage(
                    str(imagem),
                    foto_x,
                    foto_y,
                    width=foto_w,
                    height=foto_h,
                    preserveAspectRatio=True,
                    anchor="c",
                    mask="auto",
                )
            except Exception:
                c.setFillColor(PAPEL_ESCURO)
                c.rect(foto_x, foto_y, foto_w, foto_h, fill=1, stroke=0)
        else:
            c.setFillColor(PAPEL_ESCURO)
            c.rect(foto_x, foto_y, foto_w, foto_h, fill=1, stroke=0)

            c.setFillColor(CINZA)
            c.setFont(FONTES["mono"], 7)
            c.drawCentredString(
                x + largura/2,
                foto_y + foto_h/2,
                "IMAGEM NÃO DISPONÍVEL",
            )

        c.setFillColor(TINTA)
        c.setFont(FONTES["serif_bold"], 9.5)
        c.drawCentredString(x + largura/2, y + 12*mm, membro.upper())

        c.setFillColor(VERMELHO)
        c.setFont(FONTES["mono_bold"], 7.5)
        c.drawRightString(
            x + largura - 3*mm,
            y + 4.5*mm,
            f"R$ {recompensa:,.0f}".replace(",", "."),
        )
    else:
        # Página de cartazes: apenas a imagem, sem bordas de "procurado".
        if imagem and imagem.exists():
            try:
                c.drawImage(
                    str(imagem),
                    x,
                    y,
                    width=largura,
                    height=altura,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                c.setFillColor(PAPEL_ESCURO)
                c.rect(x, y, largura, altura, fill=1, stroke=0)
        else:
            c.setFillColor(PAPEL_ESCURO)
            c.rect(x, y, largura, altura, fill=1, stroke=0)

    c.restoreState()


def _pagina_cartazes(
    c,
    recompensas: Mapping[str, int],
    cartazes_dir: Path,
):
    largura, altura = A4

    _papel(c)
    _ruido_papel(c, seed=31)
    _cabecalho(c, 2, "Mural de procurados")

    c.setFillColor(TINTA)
    c.setFont(FONTES["serif_bold"], 22)
    c.drawString(18*mm, altura - 42*mm, "CARTAZES DE PROCURADOS")

    c.setFillColor(CINZA)
    c.setFont(FONTES["mono"], 7.2)
    c.drawString(
        18*mm,
        altura - 49*mm,
        "RECOMPENSAS CALCULADAS A PARTIR DO DOSSIÊ DO PERÍODO",
    )

    _linha(
        c,
        18*mm,
        altura - 53*mm,
        largura - 18*mm,
        altura - 53*mm,
        TINTA_SUAVE,
        0.5,
    )

    png_files = sorted(
        list(cartazes_dir.glob("*.png")) + list(cartazes_dir.glob("*.PNG")),
        key=lambda p: p.name.lower(),
    )

    if not png_files:
        c.setFillColor(VERMELHO)
        c.setFont(FONTES["mono_bold"], 9)
        c.drawString(18*mm, altura - 70*mm, "NENHUM PNG ENCONTRADO NO DIRETÓRIO DE CARTAZES.")
        _rodape(c, "MURAL DE PROCURADOS  •  SEM IMAGENS")
        c.showPage()
        return

    colunas = 4
    linhas = 4
    margem_x = 11*mm
    margem_y = 8*mm
    espacamento_x = 1.0*mm
    espacamento_y = 1.0*mm

    card_w = (largura - 2*margem_x - (colunas - 1) * espacamento_x) / colunas
    card_h = (altura - 76*mm - margem_y - (linhas - 1) * espacamento_y) / linhas

    topo = altura - 76*mm

    ultima_linha = (len(png_files) - 1) // colunas
    ultima_linha_itens = len(png_files) % colunas

    for idx, imagem in enumerate(png_files):
        linha = idx // colunas
        coluna = idx % colunas

        if linha >= linhas:
            break

        x = margem_x + coluna * (card_w + espacamento_x)
        y = topo - (linha + 1) * card_h - linha * espacamento_y

        if ultima_linha_itens and linha == ultima_linha:
            desloc = (colunas - ultima_linha_itens) * (card_w + espacamento_x) / 2
            x = margem_x + desloc + coluna * (card_w + espacamento_x)

        _desenhar_cartaz_mini(
            c,
            "",
            0,
            x,
            y,
            card_w,
            card_h,
            imagem,
            show_meta=False,
        )

    _carimbo(
        c,
        "ARQUIVO ATIVO",
        largura - 53*mm,
        26*mm,
        rotacao=-5,
    )

    _rodape(c, "MURAL DE PROCURADOS  •  VALORES DO PERÍODO")
    c.showPage()


# ---------------------------------------------------------------------------
# Página 3
# ---------------------------------------------------------------------------

def _pagina_dizeres(
    c,
    titulo: str,
    narrativa: str,
    data_inicio: datetime,
    data_fim: datetime,
):
    largura, altura = A4

    _papel(c)
    _ruido_papel(c, seed=73)
    _cabecalho(c, 3, "Dizeres do Inspetor")

    # Identificação temporal pequena.
    c.setFillColor(CINZA)
    c.setFont(FONTES["mono"], 6.8)
    c.drawString(
        18*mm,
        altura - 34*mm,
        f"REGISTRO: {data_inicio.strftime('%d.%m.%Y')} — {data_fim.strftime('%d.%m.%Y')}",
    )

    # Título do caso.
    y = altura - 49*mm

    y = _paragrafo(
        c,
        titulo,
        28*mm,
        y,
        largura - 56*mm,
        fonte=FONTES["serif_bold"],
        tamanho=24,
        entrelinha=27,
        cor=TINTA,
        alinhamento=TA_CENTER,
    )

    y -= 4*mm

    # Linha editorial.
    _linha(
        c,
        72*mm,
        y,
        largura - 72*mm,
        y,
        VERMELHO,
        1.2,
    )

    y -= 9*mm

    # "Entrada" de diário.
    c.setFillColor(CINZA)
    c.setFont(FONTES["mono_bold"], 7)
    c.drawString(28*mm, y, "NOTA DO INSPETOR")
    y -= 6*mm

    # Texto em coluna estreita, como página de diário.
    # Escape mínimo para o Paragraph sem alterar o conteúdo narrativo.
    texto = narrativa.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    texto = texto.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")

    y = _paragrafo(
        c,
        texto,
        28*mm,
        y,
        largura - 56*mm,
        fonte=FONTES["serif"],
        tamanho=10.7,
        entrelinha=17,
        cor=TINTA_SUAVE,
        alinhamento=TA_LEFT,
    )

    # Pequena assinatura no final.
    assinatura_y = max(28*mm, y - 9*mm)

    _linha(
        c,
        largura - 76*mm,
        assinatura_y,
        largura - 28*mm,
        assinatura_y,
        CINZA,
        0.5,
    )

    c.setFillColor(CINZA)
    c.setFont(FONTES["mono"], 6.5)
    c.drawRightString(
        largura - 28*mm,
        assinatura_y - 5*mm,
        "O INSPETOR",
    )

    _rodape(c, "DIZERES DO INSPETOR  •  FIM DO REGISTRO")
    c.showPage()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def gerar_pdf(
    *,
    titulo: str,
    narrativa: str,
    recompensas: Mapping[str, int],
    data_inicio: datetime,
    data_fim: datetime,
    cartazes_dir: Path = CARTAZES_DIR,
    output_path: Path = PDF_PATH,
) -> Path:
    """
    Gera o Diário do Inspetor completo em exatamente 3 páginas.

    Parâmetros:
        titulo:
            Título gerado pelo narrador/LLM.
        narrativa:
            Texto final gerado pelo narrador/LLM.
        recompensas:
            Mapa membro -> recompensa.
        data_inicio/data_fim:
            Intervalo coberto pelo dossiê.
        cartazes_dir:
            Diretório onde estão os PNGs dos cartazes.
        output_path:
            Caminho final do PDF.

    Retorna:
        Path do PDF gerado.
    """
    if not titulo.strip():
        raise ValueError("O título do Diário não pode estar vazio.")

    if not narrativa.strip():
        raise ValueError("A narrativa do Diário não pode estar vazia.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(
        str(output_path),
        pagesize=A4,
        pageCompression=1,
        invariant=0,
    )

    c.setTitle("Diário do Inspetor")
    c.setAuthor("SCIA")
    c.setSubject("Registro diário do Inspetor")

    _pagina_capa(c, data_inicio, data_fim)
    _pagina_cartazes(c, recompensas, Path(cartazes_dir))
    _pagina_dizeres(c, titulo, narrativa, data_inicio, data_fim)

    c.save()

    return output_path

