import asyncio
import logging

from app.controlador import Controlador


logger = logging.getLogger(__name__)

controlador = Controlador()


async def worker():

    logger.info("Worker iniciado.")

    while True:

        try:
            controlador.processar_mensagens_pendentes()
            controlador.processar_previsoes_pendentes()
            controlador.enviar_mensagens_pendentes()

        except Exception:
            logger.exception(
                "Erro no processamento das mensagens pendentes."
            )

        await asyncio.sleep(1)