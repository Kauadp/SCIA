import base64
import os
from pathlib import Path

import requests


def enviar_pdf_diario(pdf_path: str, *, destino: str | None = None):
    api_url = os.getenv("EVOLUTION_API_URL")
    api_key = os.getenv("EVOLUTION_API_KEY")
    instance = os.getenv("EVOLUTION_INSTANCE")
    grupo_jid = destino or os.getenv("DIARIO_DESTINO_JID")

    if not api_url:
        raise ValueError("EVOLUTION_API_URL não encontrada.")
    if not api_key:
        raise ValueError("EVOLUTION_API_KEY não encontrada.")
    if not instance:
        raise ValueError("EVOLUTION_INSTANCE não encontrada.")
    if not grupo_jid:
        raise ValueError("DIARIO_DESTINO_JID não encontrada. Informe o JID do grupo.")

    caminho = Path(pdf_path)

    with open(caminho, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"{api_url}/message/sendMedia/{instance}"

    payload = {
        "number": grupo_jid,
        "mediatype": "document",
        "mimetype": "application/pdf",
        "media": pdf_base64,
        "fileName": caminho.name,
        "caption": "📘 Diário do Inspetor",
    }

    response = requests.post(
        url,
        json=payload,
        headers={
            "apikey": api_key,
            "Content-Type": "application/json",
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()
