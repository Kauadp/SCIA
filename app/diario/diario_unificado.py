import sys
from pathlib import Path
import argparse
from datetime import datetime, timezone, timedelta


sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.banco.conexao import db
from app.diario.metricas import montar_dados_diario
from app.diario.gerador import preparar_contexto_episodio
from app.diario.narrativa import gerar_narrativa
from app.diario.pdf import gerar_pdf, PDF_PATH


def parse_date(s: str | None) -> datetime | None:
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            raise argparse.ArgumentTypeError(f"Formato de data inválido: {s}")


def gerar_diario(inicio: datetime | None = None, fim: datetime | None = None, output_path: str | Path | None = None) -> Path:
    fim = fim or datetime.now(timezone.utc)
    inicio = inicio or (fim - timedelta(days=7))

    print(f"Início: {inicio.isoformat()} | Fim: {fim.isoformat()}")

    previsoes = db.carregar_previsoes_periodo(inicio=inicio, fim=fim)

    if not previsoes:
        print("Nenhuma previsão encontrada no período. Abortando.")
        return Path(output_path) if output_path else PDF_PATH

    dados_diario = montar_dados_diario(previsoes)

    contexto = preparar_contexto_episodio(
        dados_diario=dados_diario,
        periodo_inicio=inicio,
        periodo_fim=fim,
    )

    resultado = gerar_narrativa(contexto)

    titulo = resultado.get("titulo", "")
    narrativa = resultado.get("narrativa", "")

    por_impostor = dados_diario.get("imposturas", {}).get("por_impostor", {})
    recompensas = {m: int(c * 100) for m, c in por_impostor.items()}

    destino = Path(output_path) if output_path else PDF_PATH

    pdf_path = gerar_pdf(
        titulo=titulo,
        narrativa=narrativa,
        recompensas=recompensas,
        data_inicio=inicio,
        data_fim=fim,
        output_path=destino,
    )

    print(f"PDF gerado: {pdf_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline unificado do Diário do Inspetor: monta dados, gera narrativa e PDF"
    )

    parser.add_argument(
        "--inicio",
        type=parse_date,
        help="Data de início (YYYY-MM-DD ou ISO). Padrão: 7 dias atrás UTC",
    )

    parser.add_argument(
        "--fim",
        type=parse_date,
        help="Data fim (YYYY-MM-DD ou ISO). Padrão: agora UTC",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho do PDF de saída. Se omitido, usa o padrão definido no módulo pdf.",
    )

    args = parser.parse_args()

    fim = args.fim or datetime.now(timezone.utc)
    inicio = args.inicio or (fim - timedelta(days=7))
    gerar_diario(inicio=inicio, fim=fim, output_path=args.output)


if __name__ == "__main__":
    main()
