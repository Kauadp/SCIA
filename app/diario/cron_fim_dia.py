import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.cartazes.semana import atualizar_cartazes
from app.diario.diario_unificado import gerar_diario
from app.diario.evolution import enviar_pdf_diario


def executar_fim_dia(
    *,
    inicio: datetime | None = None,
    fim: datetime | None = None,
    output_path: str | Path | None = None,
    enviar: bool = True,
) -> Path:
    """Atualiza os cartazes, gera o PDF do diário e envia para o JID do grupo."""
    atualizar_cartazes()

    fim = fim or datetime.now(timezone.utc)
    inicio = inicio or (fim - timedelta(days=1))

    pdf_path = gerar_diario(
        inicio=inicio,
        fim=fim,
        output_path=output_path,
    )

    if enviar:
        try:
            enviar_pdf_diario(str(pdf_path))
            print(f"PDF enviado via Evolution: {pdf_path}")
        except Exception as exc:
            print(f"Falha ao enviar PDF para a Evolution: {exc}")
            raise

    return pdf_path


def parse_date(s: str | None) -> datetime | None:
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"Formato de data inválido: {s}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Atualiza cartazes, gera o diário e envia o PDF via Evolution ao fim do dia."
    )
    parser.add_argument(
        "--inicio",
        type=parse_date,
        help="Data de início (YYYY-MM-DD ou ISO). Se omitido, usa as últimas 24 horas.",
    )
    parser.add_argument(
        "--fim",
        type=parse_date,
        help="Data fim (YYYY-MM-DD ou ISO). Se omitido, usa agora UTC.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho do PDF de saída. Se omitido, usa o padrão do módulo diario_unificado.",
    )
    parser.add_argument(
        "--skip-send",
        action="store_true",
        help="Não envia o PDF via Evolution; útil para teste local.",
    )

    args = parser.parse_args()

    fim = args.fim or datetime.now(timezone.utc)
    inicio = args.inicio or (fim - timedelta(days=1))

    executar_fim_dia(
        inicio=inicio,
        fim=fim,
        output_path=args.output,
        enviar=not args.skip_send,
    )


if __name__ == "__main__":
    main()
