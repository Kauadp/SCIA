import base64
import os
from pathlib import Path

import requests


def enviar_pdf(pdf_path: str):
    api_url = os.getenv("EVOLUTION_API_URL")
    api_key = os.getenv("EVOLUTION_API_KEY")
    instance = os.getenv("EVOLUTION_INSTANCE")
    destino = os.getenv("RELATORIO_DESTINO")

    caminho = Path(pdf_path)

    with open(caminho, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"{api_url}/message/sendMedia/{instance}"

    payload = {
        "number": destino,
        "mediatype": "document",
        "mimetype": "application/pdf",
        "media": pdf_base64,
        "fileName": caminho.name,
        "caption": "📊 Relatório semanal do SCIA",
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