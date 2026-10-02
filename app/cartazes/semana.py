import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.cartazes.recompensa import calcular_recompensa
from app.cartazes.gerador import gerar_cartaz
from app.banco.conexao import db


def atualizar_cartazes():
    dados = db.carregar_dados_recompensa()
    print(dados)

    for dado in dados:
        recompensa = calcular_recompensa(
            grupos_avaliados=dado["grupos_avaliados"],
            erro_baixo=dado["erro_baixo"],
            erro_alto=dado["erro_alto"],
            acerto_baixo=dado["acerto_baixo"],
            acerto_alto=dado["acerto_alto"],
        )

        gerar_cartaz(dado["membro"], recompensa)

        db.inserir_recompensa(
            membro=dado["membro"],
            recompensa=recompensa,
        )

        print(dado["membro"], recompensa)


if __name__ == "__main__":
    atualizar_cartazes()