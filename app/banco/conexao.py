from sqlalchemy import create_engine, text
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import json
logger = logging.getLogger(__name__)


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
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "membro": membro,
                    "data_hora": data_hora,
                    "mensagem": mensagem,
                }
            )

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

    def inserir_previsao(
        self,
        mensagem_id: int,
        mensagem: str,
        autor_real: str,
        autor_predito: str,
        probabilidade: float,
        categoria: str,
        features: dict,
        status_bot: str = "PENDENTE"
    ):
        query = text("""
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
                    "features": json.dumps(features, ensure_ascii=False),
                    "status_bot": status_bot,
                }
            )

            return resultado.scalar_one_or_none()

    def marcar_mensagem_processada(self, mensagem_id: int):
        query = text("""
            UPDATE mensagens_raw
            SET processada = TRUE
            WHERE id = :mensagem_id
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {"mensagem_id": mensagem_id}
            )

    def carregar_previsoes_pendentes(self):

        query = text("""
            SELECT
                id,
                categoria
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
        """)

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "previsao_id": previsao_id,
                    "categoria": categoria,
                    "personalidade": personalidade,
                    "texto": texto,
                }
            )

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
            SELECT
                id,
                texto
            FROM mensagens_bot
            WHERE status = 'PENDENTE_ENVIO'
            ORDER BY id
            LIMIT 10
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

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

db = DatabaseManager(connection_string=os.getenv("db_uri"))