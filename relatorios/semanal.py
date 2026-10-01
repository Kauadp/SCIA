from dados import db
from metricas import calcular_metricas
from graficos import gerar_graficos
from pdf import construir_pdf
from evolution import enviar_pdf

df_raw = db.carregar_mensagens_raw()
df_prev = db.carregar_previsoes()
df_bot = db.carregar_mensagens_bot()

metricas = calcular_metricas(
    df_raw,
    df_prev,
    df_bot,
)

graficos = gerar_graficos(
    metricas,
    df_prev,
)

construir_pdf(
    metricas,
    graficos,
    "relatorios/relatorio_semanal.pdf"
)

enviar_pdf("relatorios/relatorio_semanal.pdf")