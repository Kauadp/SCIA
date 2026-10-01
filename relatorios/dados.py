from pathlib import Path
import logging
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


logger = logging.getLogger(__name__)


class DatabaseManager:

    def __init__(self, connection_string: str):
        """Inicializa a conexão com o Banco de Dados."""
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

    def carregar_mensagens_raw(self):
        query = text("""
            SELECT *
            FROM mensagens_raw
            ORDER BY id
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)

    def carregar_previsoes(self):
        query = text("""
            SELECT *
            FROM previsoes
            ORDER BY id
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)

    def carregar_mensagens_bot(self):
        query = text("""
            SELECT *
            FROM mensagens_bot
            ORDER BY id
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)



load_dotenv()

db = DatabaseManager(
    connection_string=os.getenv("db_uri")
)