from app.banco.conexao import db
from app.modelo.preditor import prever_mensagem
from app.llm import gerar_mensagem
from app.evolution.cliente import enviar_texto
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def classificar_previsao(
    autor_real: str,
    autor_predito: str,
    probabilidade: float,
) -> str | None:

    if probabilidade < 0.70:
        return None

    if probabilidade < 0.80:
        nivel = "BAIXO"
    elif probabilidade < 0.90:
        nivel = "MEDIO"
    else:
        nivel = "ALTO"

    tipo = "ACERTO" if autor_real == autor_predito else "ERRO"

    return f"{tipo}_{nivel}"

class Controlador:

    def processar_mensagem(self, mensagem: dict):
        try:
            db.inserir_mensagem(
                membro=mensagem["membro"],
                data_hora=mensagem["data_hora"],
                mensagem=mensagem["mensagem"],
            )
            logger.info("Mensagem inserida no banco com sucesso.")
        except Exception as e:
            logger.error(
                f"Erro ao inserir os dados: {str(e)}"
            )
            raise
        
    def processar_mensagens_pendentes(self):

        mensagens = db.carregar_mensagens_nao_processadas()

        logger.info(
            f"{len(mensagens)} mensagens pendentes encontradas."
        )

        for _, mensagem in mensagens.iterrows():

            mensagem_id = int(mensagem["id"])

            try:
                dados_mensagem = {
                    "data_hora": mensagem["data_hora"],
                    "membro": mensagem["membro"],
                    "mensagem": mensagem["mensagem"],
                }

                resultado = prever_mensagem(dados_mensagem)

                if resultado is None:
                    logger.warning(
                        f"Não foi possível prever mensagem {mensagem_id}."
                    )
                    continue

                features = {
                    "dia": resultado["dia"],
                    "hora": resultado["hora"],
                    "cluster": resultado["cluster"],
                    "caracter_por_mensagem": resultado["caracter_por_mensagem"],
                }


                categoria = classificar_previsao(
                    autor_real=mensagem["membro"],
                    autor_predito=resultado["autor_previsto"],
                    probabilidade=resultado["probabilidade"],
                )

                db.inserir_previsao(
                    mensagem_id=mensagem_id,
                    mensagem=mensagem["mensagem"],
                    autor_real=mensagem["membro"],
                    autor_predito=resultado["autor_previsto"],
                    probabilidade=resultado["probabilidade"],
                    categoria=categoria,
                    features=features,
                )

                db.marcar_mensagem_processada(mensagem_id)

                logger.info(
                    f"Mensagem {mensagem_id} processada. "
                    f"Real: {mensagem['membro']} | "
                    f"Previsto: {resultado['autor_previsto']} | "
                    f"Confiança: {resultado['probabilidade']:.2%}"
                )

            except Exception:
                logger.exception(
                    f"Erro ao processar mensagem {mensagem_id}."
                )

    def processar_previsoes_pendentes(self):

        previsoes = db.carregar_previsoes_pendentes()

        logger.info(
            f"{len(previsoes)} previsões pendentes encontradas."
        )

        for _, previsao in previsoes.iterrows():

            previsao_id = int(previsao["id"])

            try:
                texto, personalidade = gerar_mensagem(
                    categoria=previsao["categoria"]
                )

                db.inserir_mensagem_bot(
                    previsao_id=previsao_id,
                    categoria=previsao["categoria"],
                    personalidade=personalidade,
                    texto=texto,
                )

                db.marcar_previsao_processada(previsao_id)

                logger.info(
                    f"Mensagem do SCIA gerada para previsão "
                    f"{previsao_id} | "
                    f"Categoria: {previsao['categoria']} | "
                    f"Personalidade: {personalidade}"
                )

            except Exception:
                logger.exception(
                    f"Erro ao gerar mensagem para previsão "
                    f"{previsao_id}."
                )

    def enviar_mensagens_pendentes(self):

        mensagens = db.carregar_mensagens_bot_pendentes()

        logger.info(
            f"{len(mensagens)} mensagens aguardando envio."
        )

        for _, mensagem in mensagens.iterrows():

            mensagem_id = int(mensagem["id"])

            try:

                enviar_texto(
                    numero="120363025949767428@g.us",
                    texto=mensagem["texto"],
                )

                db.marcar_mensagem_enviada(mensagem_id)

                logger.info(
                    f"Mensagem {mensagem_id} enviada com sucesso."
                )

            except Exception:

                db.marcar_mensagem_erro(mensagem_id)

                logger.exception(
                    f"Erro ao enviar mensagem {mensagem_id}."
                )