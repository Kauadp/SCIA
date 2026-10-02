import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.banco.conexao import db
from app.diario.metricas import (
    montar_dados_diario
)
from app.diario.gerador import preparar_contexto_episodio
from datetime import datetime, timezone

inicio = datetime(2026, 10, 1, tzinfo=timezone.utc)
fim=datetime(2026, 10, 3, tzinfo=timezone.utc)

print(f"Período Inicial: {inicio}\nPeríodo Final: {fim}")

previsoes = db.carregar_previsoes_periodo(
    inicio=inicio,
    fim=fim,
)

dados_diario = montar_dados_diario(previsoes)

contexto = preparar_contexto_episodio(
    dados_diario=dados_diario,
    periodo_inicio=inicio,
    periodo_fim=fim,
)

from app.diario.narrativa import gerar_narrativa

resultado = gerar_narrativa(contexto)

print("=== TÍTULO ===")
print(resultado["titulo"])

print()
print("=== NARRATIVA ===")
print(resultado["narrativa"])