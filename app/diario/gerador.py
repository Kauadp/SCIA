from datetime import datetime, timezone
from zoneinfo import ZoneInfo

INTENSIDADE_POR_CATEGORIA = {
    "ERRO_BAIXO": "fraco",
    "ERRO_ALTO": "forte",
}

FUSO_LOCAL = ZoneInfo("America/Sao_Paulo")
 
 
def formatar_data(data):
    """
    Converte datetime (UTC ou qualquer timezone) para o horário local do
    grupo, em formato legível. Isso importa para o Inspetor: "madrugada"
    só deve ser usado se for realmente madrugada no horário local, não
    no horário em que o banco guarda o timestamp.
    """
 
    if data.tzinfo is None:
        data = data.replace(tzinfo=timezone.utc)
 
    data_local = data.astimezone(FUSO_LOCAL)
 
    return data_local.strftime("%d/%m/%Y às %H:%M")



def preparar_contexto_episodio(
    dados_diario,
    periodo_inicio,
    periodo_fim,
):
    """
    Transforma os dados objetivos do episódio em um contexto textual
    estruturado para uso posterior pelo gerador narrativo.

    IMPORTANTE: este contexto nunca deve conter "ERRO_BAIXO", "ERRO_ALTO",
    "confiança" ou qualquer porcentagem — essas informações já chegam
    traduzidas em "intensidade_do_sinal" (fraco/forte) antes de virar
    texto. O bloco de estatísticas agregadas (KPIs técnicos) foi removido
    de propósito: o próprio SYSTEM_PROMPT já instrui o narrador a nunca
    usar métricas agregadas na narrativa, então mandá-las aqui só criava
    superfície de vazamento sem nenhum ganho.
    """

    imposturas = dados_diario["imposturas"]
    detalhes = dados_diario["detalhes_imposturas"]

    linhas = []

    linhas.append("=== DIÁRIO DO INSPETOR ===")
    linhas.append("")

    linhas.append("=== PERÍODO ===")
    linhas.append(f"Início: {periodo_inicio.strftime('%d/%m/%Y')}")
    linhas.append(f"Fim: {periodo_fim.strftime('%d/%m/%Y')}")
    linhas.append("")

    linhas.append("=== PADRÕES DE IMPOSTURA ===")

    if imposturas["por_impostor"]:
        linhas.append("Imposturas por membro:")
        for membro, quantidade in imposturas["por_impostor"].items():
            linhas.append(f"- {membro}: {quantidade}")
    else:
        linhas.append("- Nenhuma impostura detectada.")

    linhas.append("")

    if imposturas["identidades_imitadas"]:
        linhas.append("Identidades mais imitadas:")
        for membro, quantidade in imposturas["identidades_imitadas"].items():
            linhas.append(f"- {membro}: {quantidade}")

    linhas.append("")

    if imposturas["pares_impostura"]:
        linhas.append("Pares de impostura:")
        for par, quantidade in imposturas["pares_impostura"].items():
            linhas.append(f"- {par}: {quantidade}")

    linhas.append("")

    linhas.append("=== OCORRÊNCIAS DE IMPOSTURA ===")

    if not detalhes:
        linhas.append("Nenhuma ocorrência de impostura.")
    else:
        for i, detalhe in enumerate(detalhes, start=1):
            intensidade = INTENSIDADE_POR_CATEGORIA[detalhe["categoria"]]

            linhas.append("")
            linhas.append(f"[OCORRÊNCIA {i}]")
            linhas.append(f"Autor real: {detalhe['autor_real']}")
            linhas.append(f"Identidade detectada: {detalhe['autor_predito']}")
            linhas.append(f"Intensidade do sinal: {intensidade}")
            linhas.append(f"Data: {formatar_data(detalhe['data_hora'])}")
            linhas.append(f"Mensagem: {detalhe['mensagem']}")

    return "\n".join(linhas)