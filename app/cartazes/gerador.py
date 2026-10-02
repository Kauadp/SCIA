from pathlib import Path
import unicodedata

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
ARTIFACTS_DIR = BASE_DIR / "cartazes_atualizados"

RECOMPENSA_X = 590
RECOMPENSA_Y = 1230

TAMANHO_FONTE = 90

FONTE = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

def remover_acentos(texto):
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def gerar_cartaz(membro: str, recompensa: int) -> Path:
    """
    Gera o cartaz de um membro usando seu template fixo
    e sobrescrevendo apenas o valor da recompensa.
    """

    membro = remover_acentos(membro).lower()
    
    template_path = TEMPLATES_DIR / f"{membro}.png"

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template não encontrado: {template_path}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    imagem = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(imagem)

    fonte = ImageFont.truetype(FONTE, TAMANHO_FONTE)

    texto = f"{recompensa:,.0f}".replace(",", ".")

    caixa = draw.textbbox((0, 0), texto, font=fonte)
    largura = caixa[2] - caixa[0]
    altura = caixa[3] - caixa[1]

    x = RECOMPENSA_X - largura / 2
    y = RECOMPENSA_Y - altura / 2

    draw.text(
        (x, y),
        texto,
        font=fonte,
        fill="black",
    )

    output_path = ARTIFACTS_DIR / f"{membro}.png"
    imagem.save(output_path)

    return output_path


if __name__ == "__main__":
    caminho = gerar_cartaz(
        membro="kaua",
        recompensa=18_500,
    )

    print(f"Cartaz gerado: {caminho}")