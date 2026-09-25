from datetime import datetime
from .maps import MAP_MEMBRO
from zoneinfo import ZoneInfo

GROUP_ID = "120363025949767428@g.us"

def processar_webhook(payload):
    data = payload.get("data")
    if not data:
        return None

    key = data.get("key", {})
    if key.get("remoteJid") != GROUP_ID:
        return None

    message = data.get("message")
    if not message:
        return None

    mensagem_texto = (
        message.get("conversation")
        or message.get("extendedTextMessage", {}).get("text")
        or message.get("imageMessage", {}).get("caption")
        or message.get("videoMessage", {}).get("caption")
    )

    if not mensagem_texto:
        return None

    timestamp = data["messageTimestamp"]
    
    data_hora = datetime.fromtimestamp(
        timestamp,
        tz=ZoneInfo("America/Sao_Paulo")
    ).isoformat()

    return {
        "data_hora": data_hora,
        "membro": MAP_MEMBRO.get(key.get("participant")),
        "mensagem": mensagem_texto,
        "message_id": key.get("id"),
    }