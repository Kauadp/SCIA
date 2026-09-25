import asyncio
import logging

from fastapi import FastAPI, Request

from app.controlador import Controlador
from app.webhook.parser import processar_webhook
from app.worker import worker


logger = logging.getLogger(__name__)

app = FastAPI()

controlador = Controlador()

@app.on_event("startup")
async def iniciar_worker():

    asyncio.create_task(worker())

    logger.info("Worker iniciado junto com a aplicação.")


@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
        mensagem = processar_webhook(payload)

        if mensagem is not None:
            controlador.processar_mensagem(mensagem)

    except Exception as e:
        print(f"Erro ao processar payload: {e}")

    return {"status": "ok"}

