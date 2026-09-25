import os
import requests

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE")

if not EVOLUTION_API_URL:
    raise ValueError("EVOLUTION_API_URL não encontrada.")

if not EVOLUTION_API_KEY:
    raise ValueError("EVOLUTION_API_KEY não encontrada.")

if not EVOLUTION_INSTANCE:
    raise ValueError("EVOLUTION_INSTANCE não encontrada.")


def enviar_texto(
    numero: str,
    texto: str,
) -> dict:

    url = (
        f"{EVOLUTION_API_URL}"
        f"/message/sendText/{EVOLUTION_INSTANCE}"
    )

    headers = {
        "Content-Type": "application/json",
        "apikey": EVOLUTION_API_KEY,
    }

    payload = {
        "number": numero,
        "text": texto,
    }

    resposta = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    resposta.raise_for_status()

    return resposta.json()