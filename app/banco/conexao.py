from sqlalchemy import create_engine, text
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import json
logger = logging.getLogger(__name__)

COOLDOWN_MINUTOS = 5

class DatabaseManager:

    def __init__(self, connection_string: str):
        """Inicializa a conexão com o Banco de Dados através do SQLAlchemy Engine."""
        self.connection_string = connection_string

        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )

            logger.info("Engine do Banco de Dados inicializado com sucesso.")

        except Exception as e:
            logger.error(
                f"Erro ao criar o Engine do Banco de Dados: {str(e)}"
            )
            raise

    def inserir_mensagem(
        self,
        membro: str,
        data_hora,
        mensagem: str
    ):
        query = text("""
            INSERT INTO mensagens_raw (
                membro,
                data_hora,
                mensagem,
                processada
            )
            VALUES (
                :membro,
                :data_hora,
                :mensagem,
                FALSE
            )
            RETURNING id
        """)

        with self.engine.begin() as conn:
            resultado = conn.execute(
                query,
                {
                    "membro": membro,
                    "data_hora": data_hora,
                    "mensagem": mensagem,
                }
            )

            return resultado.scalar_one()

    def marcar_mensagem_processada(self, mensagem_id: int):
        query = text("""
            UPDATE mensagens_raw
            SET processada = TRUE
            WHERE id = :id
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {"id": mensagem_id}
            )

    def marcar_mensagens_processadas(self, mensagem_ids: list[int]):
        if not mensagem_ids:
            return

        query = text("""
            UPDATE mensagens_raw
            SET processada = TRUE
            WHERE id = ANY(:mensagem_ids)
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "mensagem_ids": mensagem_ids
                }
            )

    def carregar_mensagens_nao_processadas(self):
        query = text("""
            SELECT
                id,
                membro,
                data_hora,
                mensagem,
                processada
            FROM mensagens_raw
            WHERE processada = FALSE
            ORDER BY id
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)

    def carregar_mensagens_v2_pendentes(self):
        query = text("""
            SELECT
                id,
                membro,
                data_hora,
                mensagem
            FROM mensagens_raw
            WHERE processada = FALSE
            ORDER BY membro, data_hora, id
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)

    def carregar_mensagens_raw(self): # Para testes
            query = text("""
                SELECT
                    id,
                    membro,
                    data_hora,
                    mensagem,
                    processada
                FROM mensagens_raw
                ORDER BY id
            """)
    
            with self.engine.begin() as conn:
                return pd.read_sql(query, conn)

    def inserir_previsao(
        self,
        mensagem_id: int,
        mensagem: str,
        autor_real: str,
        autor_predito: str,
        probabilidade: float,
        categoria: str | None,
        features: dict,
        status_bot: str = "PENDENTE"
    ):

        query = text("""
            WITH nova_previsao AS (
                INSERT INTO previsoes (
                    mensagem_id,
                    mensagem,
                    autor_real,
                    autor_predito,
                    probabilidade,
                    categoria,
                    features,
                    status_bot
                )
                VALUES (
                    :mensagem_id,
                    :mensagem,
                    :autor_real,
                    :autor_predito,
                    :probabilidade,
                    :categoria,
                    CAST(:features AS JSONB),
                    :status_bot
                )
                ON CONFLICT (mensagem_id) DO NOTHING
                RETURNING id
            )
            SELECT id FROM nova_previsao

            UNION ALL

            SELECT id
            FROM previsoes
            WHERE mensagem_id = :mensagem_id

            LIMIT 1
        """)

        with self.engine.begin() as conn:
            resultado = conn.execute(
                query,
                {
                    "mensagem_id": mensagem_id,
                    "mensagem": mensagem,
                    "autor_real": autor_real,
                    "autor_predito": autor_predito,
                    "probabilidade": probabilidade,
                    "categoria": categoria,
                    "features": json.dumps(features),
                    "status_bot": status_bot
                }
            )

            return resultado.scalar_one_or_none()

    def carregar_previsoes_pendentes(self):

        query = text("""
            SELECT
                id,
                categoria,
                autor_real,
                autor_predito,
                probabilidade
            FROM previsoes
            WHERE categoria IS NOT NULL
            AND status_bot = 'PENDENTE'
            ORDER BY id
            LIMIT 10
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)

    def inserir_mensagem_bot(
        self,
        previsao_id: int,
        categoria: str,
        personalidade: str,
        texto: str,
    ):
        query = text("""
            INSERT INTO mensagens_bot (
                previsao_id,
                categoria,
                personalidade,
                texto,
                status
            )
            VALUES (
                :previsao_id,
                :categoria,
                :personalidade,
                :texto,
                'PENDENTE_ENVIO'
            )
            ON CONFLICT (previsao_id) DO NOTHING
            RETURNING id
        """)

        with self.engine.begin() as conn:
            resultado = conn.execute(
                query,
                {
                    "previsao_id": previsao_id,
                    "categoria": categoria,
                    "personalidade": personalidade,
                    "texto": texto,
                }
            )

            return resultado.scalar_one_or_none()

    def marcar_previsao_processada(self, previsao_id: int):

        query = text("""
            UPDATE previsoes
            SET status_bot = 'PROCESSADO'
            WHERE id = :previsao_id
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "previsao_id": previsao_id
                }
            )

    def carregar_mensagens_bot_pendentes(self):

        query = text("""
            SELECT *
            FROM mensagens_bot
            WHERE status = 'PENDENTE_ENVIO'
                AND gerado_em <= NOW() - INTERVAL '30 seconds'
            ORDER BY id;
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)

    def marcar_mensagem_enviada(self, mensagem_id: int):

        query = text("""
            UPDATE mensagens_bot
            SET
                status = 'ENVIADO',
                enviado_em = NOW()
            WHERE id = :mensagem_id
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {"mensagem_id": mensagem_id}
            )

    def marcar_mensagem_erro(self, mensagem_id: int):

        query = text("""
            UPDATE mensagens_bot
            SET status = 'ERRO_ENVIO'
            WHERE id = :mensagem_id
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {"mensagem_id": mensagem_id}
            )

    def pessoa_em_cooldown(self, autor_real: str) -> bool:
        query = text("""
            SELECT EXISTS (
                SELECT 1
                FROM mensagens_bot mb
                JOIN previsoes p
                    ON p.id = mb.previsao_id
                WHERE p.autor_real = :autor_real
                AND (
                    (
                        mb.status = 'ENVIADO'
                        AND mb.enviado_em >= NOW() - (
                            :cooldown * INTERVAL '1 minute'
                        )
                    )
                    OR
                    mb.status = 'PENDENTE_ENVIO'
                )
            )
        """)

        with self.engine.begin() as conn:
            return conn.execute(
                query,
                {
                    "autor_real": autor_real,
                    "cooldown": COOLDOWN_MINUTOS,
                }
            ).scalar()

    def carregar_historico_membro(self, membro: str):

        query = text("""
            SELECT
                autor_real,
                autor_predito,
                COUNT(*) AS quantidade
            FROM previsoes
            WHERE autor_real = :membro
            AND probabilidade >= 0.80
            GROUP BY autor_real, autor_predito
            ORDER BY quantidade DESC
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(
                query,
                conn,
                params={"membro": membro},
            )

    def processar_scan(self, membro: str) -> str:

        historico = db.carregar_historico_membro(membro)

        if historico.empty:
            return (
                f"🕵️ *FICHA SCIA — {membro.upper()}*\n\n"
                f"📭 O Dispositivo ainda não possui dados suficientes "
                f"sobre este indivíduo."
            )

        ficha = self.gerar_ficha_membro(historico)

        mensagem = (
            f"🕵️ *FICHA SCIA — {membro.upper()}*\n\n"
            f"📊 *Mensagens analisadas:* {ficha['total']}\n"
            f"✅ *Identidade confirmada:* {ficha['acertos']}\n"
            f"🚨 *Identidade contestada:* {ficha['erros']}\n"
            f"🎭 *Taxa de impostura:* "
            f"{ficha['taxa_impostura']:.1%}\n"
        )

        if ficha["principal_confusao"] is not None:
            mensagem += (
                f"\n🎯 *Identidade mais atribuída:* "
                f"{ficha['principal_confusao']}"
            )

        return mensagem

    def carregar_dados_recompensa(self):
        query = """
            SELECT
                autor_real AS membro,
                COUNT(*) AS grupos_avaliados,

                COUNT(*) FILTER (
                    WHERE categoria = 'ERRO_BAIXO'
                ) AS erro_baixo,

                COUNT(*) FILTER (
                    WHERE categoria = 'ERRO_ALTO'
                ) AS erro_alto,

                COUNT(*) FILTER (
                    WHERE categoria = 'ACERTO_BAIXO'
                ) AS acerto_baixo,

                COUNT(*) FILTER (
                    WHERE categoria = 'ACERTO_ALTO'
                ) AS acerto_alto

            FROM previsoes
            GROUP BY autor_real
            ORDER BY autor_real;
        """

        with self.engine.connect() as conn:
            resultado = conn.execute(text(query))

            return [dict(row._mapping) for row in resultado]

    def inserir_recompensa(
        self,
        membro: str,
        recompensa: int,
    ):
        query = """
            INSERT INTO recompensas (
                membro,
                recompensa
            )
            VALUES (
                :membro,
                :recompensa
            )
        """

        with self.engine.begin() as conn:
            conn.execute(
                text(query),
                {
                    "membro": membro,
                    "recompensa": recompensa,
                }
            )

    def carregar_previsoes_periodo(
        self,
        inicio,
        fim,
    ):
        query = """
            SELECT
                autor_real,
                autor_predito,
                probabilidade,
                categoria,
                mensagem,
                processado_em
            FROM previsoes
            WHERE processado_em >= :inicio
            AND processado_em < :fim
            ORDER BY processado_em;
        """

        with self.engine.connect() as conn:
            resultado = conn.execute(
                text(query),
                {
                    "inicio": inicio,
                    "fim": fim,
                }
            )

            return [dict(row._mapping) for row in resultado]

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

db = DatabaseManager(connection_string=os.getenv("db_uri"))