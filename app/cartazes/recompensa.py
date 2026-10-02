import math


PESO_ERRO_BAIXO = 500
PESO_ERRO_ALTO = 1_000

PESO_ACERTO_BAIXO = -50
PESO_ACERTO_ALTO = -150


def calcular_recompensa(
    grupos_avaliados: int,
    erro_baixo: int,
    erro_alto: int,
    acerto_baixo: int,
    acerto_alto: int,
) -> int:

    pontos_culpa = (
        PESO_ERRO_BAIXO * erro_baixo
        + PESO_ERRO_ALTO * erro_alto
        + PESO_ACERTO_BAIXO * acerto_baixo
        + PESO_ACERTO_ALTO * acerto_alto
    )

    if pontos_culpa <= 0:
        return 0

    peso_atividade = 1 + math.log1p(grupos_avaliados)

    recompensa = pontos_culpa * peso_atividade

    return round(recompensa)