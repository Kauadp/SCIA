from app.banco.conexao import db
from app.modelo.preditor import prever_autor
from app.llm import gerar_mensagem
from app.evolution.cliente import enviar_texto
import logging
import pandas as pd

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

    if probabilidade < 0.80:
        return None

    if probabilidade < 0.90:
        nivel = "BAIXO"
    else:
        nivel = "ALTO"

    tipo = "ACERTO" if autor_real == autor_predito else "ERRO"

    return f"{tipo}_{nivel}"

def montar_mensagem_deteccao(texto_llm: str) -> str:
    return (
        f"🚨 *INSPETOR DETECTOU* 🚨\n\n"
        f"🕵️ *INSPETOR DIZ:*\n\n"
        f"\"{texto_llm}\""
    )

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

        mensagens = db.carregar_mensagens_v2_pendentes()

        logger.info(
            f"{len(mensagens)} mensagens pendentes encontradas."
        )

        if mensagens.empty:
            return

        mensagens["data_hora"] = pd.to_datetime(
            mensagens["data_hora"]
        )

        mensagens["grupo_5"] = (
            mensagens
            .groupby("membro")
            .cumcount() // 5
        )

        grupos = list(
            mensagens.groupby(
                ["membro", "grupo_5"],
                sort=False
            )
        )

        logger.warning(
            "DEBUG GRUPOS: %s",
            [
                (
                    membro,
                    grupo_id,
                    len(grupo),
                    grupo["id"].astype(int).tolist()
                )
                for (membro, grupo_id), grupo in grupos
            ]
        )

        for (membro, grupo_id), grupo in grupos:

            if len(grupo) < 5:
                logger.info(
                    f"Grupo incompleto ignorado. "
                    f"Membro: {membro} | "
                    f"Mensagens: {len(grupo)}/5"
                )
                continue

            ids = grupo["id"].astype(int).tolist()

            try:

                dados_grupo = grupo[
                    ["data_hora", "membro", "mensagem"]
                ].copy()

                resultados = prever_autor(dados_grupo)

                if resultados is None or len(resultados) == 0:
                    logger.warning(
                        f"Não foi possível prever grupo "
                        f"{grupo_id} do membro {membro}."
                    )
                    continue

                resultado = resultados[0]

                autor_real = grupo["membro"].iloc[0]
                autor_predito = resultado["autor_previsto"]
                probabilidade = resultado["probabilidade"]

                features = {
                    "dia_inicio": resultado["dia_inicio"],
                    "dia_fim": resultado["dia_fim"],
                    "hora_inicio": resultado["hora_inicio"],
                    "hora_fim": resultado["hora_fim"],
                    "log_duracao": resultado["log_duracao"],
                    "qtd_palavras": resultado["qtd_palavras"],
                    "qtd_caracteres": resultado["qtd_caracteres"],
                    "caracter_por_palavra": resultado["caracter_por_palavra"],
                }

                categoria = classificar_previsao(
                    autor_real=autor_real,
                    autor_predito=autor_predito,
                    probabilidade=probabilidade,
                )

                mensagem_consolidada = " ".join(
                    grupo["mensagem"]
                    .fillna("")
                    .astype(str)
                )

                # Primeiro ID do grupo como âncora da previsão
                mensagem_id = ids[0]

                previsao_id = db.inserir_previsao(
                    mensagem_id=mensagem_id,
                    mensagem=mensagem_consolidada,
                    autor_real=autor_real,
                    autor_predito=autor_predito,
                    probabilidade=probabilidade,
                    categoria=categoria,
                    features=features,
                )

                if previsao_id is None:
                    logger.error(
                        f"Não foi possível obter a previsão "
                        f"do grupo {grupo_id}. "
                        f"IDs: {ids}"
                    )
                    continue

                db.marcar_mensagens_processadas(ids)
                
                if categoria is None:
                    db.marcar_previsao_processada(previsao_id)

                logger.info(
                    f"Grupo processado com sucesso. "
                    f"Membro: {autor_real} | "
                    f"IDs: {ids} | "
                    f"Previsto: {autor_predito} | "
                    f"Confiança: {probabilidade:.2%} | "
                    f"Categoria: {categoria}"
                )

            except Exception:
                logger.exception(
                    f"Erro ao processar grupo {grupo_id} "
                    f"do membro {membro}. "
                    f"IDs: {ids}"
                )

    def processar_previsoes_pendentes(self):

        previsoes = db.carregar_previsoes_pendentes()

        logger.info(
            f"{len(previsoes)} previsões pendentes encontradas."
        )

        for _, previsao in previsoes.iterrows():

            previsao_id = int(previsao["id"])

            categoria = previsao["categoria"]

            if categoria.startswith("ACERTO"):
                db.marcar_previsao_processada(previsao_id)

                logger.info(
                    f"Previsão {previsao_id} classificada como "
                    f"{categoria}. Nenhuma mensagem será gerada."
                )

                continue

            autor_real = previsao["autor_real"]

            if db.pessoa_em_cooldown(autor_real):
                db.marcar_previsao_processada(previsao_id)

                logger.info(
                    f"Previsão {previsao_id} ignorada por cooldown. "
                    f"Autor: {autor_real}"
                )

                continue

            try:
                texto, personalidade = gerar_mensagem(
                    categoria=categoria,
                    autor_real=autor_real,
                    autor_predito=previsao["autor_predito"],
                )

                texto_final = montar_mensagem_deteccao(
                    texto_llm=texto
                )

                db.inserir_mensagem_bot(
                    previsao_id=previsao_id,
                    categoria=categoria,
                    personalidade=personalidade,
                    texto=texto_final,
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

    def gerar_ficha_membro(self, historico) -> dict:

        total = int(historico["quantidade"].sum())

        acertos = int(
            historico.loc[
                historico["autor_real"] == historico["autor_predito"],
                "quantidade"
            ].sum()
        )

        erros = total - acertos
        taxa_impostura = erros / total if total > 0 else 0

        confundido_com = historico[
            historico["autor_real"] != historico["autor_predito"]
        ].copy()

        confundido_com = confundido_com.sort_values(
            "quantidade",
            ascending=False
        )

        top_3 = confundido_com.head(3)

        if taxa_impostura < 0.10:
            titulo = "CIDADÃO EXEMPLAR"
            perfil = (
                "O indivíduo apresenta histórico "
                "praticamente livre de falsificação de identidade.\n\n"
                "O inspetor não vê motivos relevantes para suspeita."
            )

        elif taxa_impostura < 0.20:
            titulo = "PEQUENO GOLPISTA"
            perfil = (
                "O indivíduo apresenta histórico "
                "moderado de falsificação de identidade.\n\n"
                "O inspetor recomenda vigilância."
            )

        elif taxa_impostura < 0.35:
            titulo = "SUSPEITO RECORRENTE"
            perfil = (
                "O indivíduo apresenta histórico "
                "considerável de falsificação de identidade.\n\n"
                "O inspetor recomenda atenção redobrada."
            )

        else:
            titulo = "AMEAÇA À IDENTIDADE ALHEIA"
            perfil = (
                "O indivíduo apresenta histórico grave "
                "de falsificação de identidade.\n\n"
                "O inspetor recomenda vigilância máxima."
            )

        return {
            "total": total,
            "acertos": acertos,
            "erros": erros,
            "taxa_impostura": taxa_impostura,
            "top_3": top_3,
            "titulo": titulo,
            "perfil": perfil,
        }

    def processar_scan(self, membro: str) -> str:

        historico = db.carregar_historico_membro(membro)

        if historico.empty:
            return (
                "╔══════════════════════════╗\n"
                "     🕵️ FICHA CRIMINAL\n"
                "╚══════════════════════════╝\n\n"
                f"👤 SUSPEITO\n"
                f"{membro.upper()}\n\n"
                "📭 O Dispositivo ainda não possui dados suficientes "
                "sobre este indivíduo."
            )

        ficha = self.gerar_ficha_membro(historico)

        total = ficha["total"]
        acertos = ficha["acertos"]
        erros = ficha["erros"]
        taxa = ficha["taxa_impostura"]

        percentual_acerto = acertos / total if total > 0 else 0

        mensagem = (
            "╔══════════════════════════╗\n"
            "     🕵️ FICHA CRIMINAL\n"
            "╚══════════════════════════╝\n\n"

            "👤 SUSPEITO\n"
            f"{membro.upper()}\n\n"

            "🔎 INVESTIGAÇÕES\n"
            f"Mensagens analisadas: {total}\n\n"

            "🟢 IDENTIDADE CONFIRMADA\n"
            f"{acertos} mensagens\n"
            f"{percentual_acerto:.1%}\n\n"

            "🔴 IMPOSTOR DETECTADO\n"
            f"{erros} mensagens\n"
            f"{taxa:.1%}\n\n"

            "⚠️ ÍNDICE DE SUSPEITA\n"
            f"{taxa:.1%}\n\n"

            "🏷️ TÍTULO\n"
            f"{ficha['titulo']}\n\n"

            "──────────────────────────\n\n"

            "🎭 QUEM "
            f"{membro.upper()} MAIS SE PASSOU\n\n"
        )

        if ficha["top_3"].empty:
            mensagem += "Nenhum doppelgänger identificado.\n"
        else:
            medalhas = ["🥇", "🥈", "🥉"]

            for i, (_, linha) in enumerate(
                ficha["top_3"].iterrows()
            ):
                mensagem += (
                    f"{medalhas[i]} {linha['autor_predito'].upper()}"
                    f" — {int(linha['quantidade'])} vezes\n"
                )

        mensagem += (
            "\n──────────────────────────\n\n"

            "📊 PERFIL DO SUSPEITO\n\n"
            f"{ficha['perfil']}"
        )

        return mensagem

    def enviar_mensagem_scan(self, texto: str):

        enviar_texto(
            numero="120363025949767428@g.us",
            texto=texto,
        )