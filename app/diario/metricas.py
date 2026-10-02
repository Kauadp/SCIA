from collections import Counter


def calcular_kpis(previsoes):
    """
    Calcula as principais métricas objetivas do episódio
    a partir das previsões do período.
    """

    grupos_avaliados = len(previsoes)

    grupos_acima_080 = sum(
        previsao["probabilidade"] >= 0.80
        for previsao in previsoes
    )

    cobertura_080 = (
        grupos_acima_080 / grupos_avaliados
        if grupos_avaliados > 0
        else 0
    )

    erro_baixo = sum(
        previsao["categoria"] == "ERRO_BAIXO"
        for previsao in previsoes
    )

    erro_alto = sum(
        previsao["categoria"] == "ERRO_ALTO"
        for previsao in previsoes
    )

    acerto_baixo = sum(
        previsao["categoria"] == "ACERTO_BAIXO"
        for previsao in previsoes
    )

    acerto_alto = sum(
        previsao["categoria"] == "ACERTO_ALTO"
        for previsao in previsoes
    )

    imposturas_total = erro_baixo + erro_alto
    acertos_total = acerto_baixo + acerto_alto

    taxa_impostura = (
        imposturas_total / grupos_acima_080
        if grupos_acima_080 > 0
        else 0
    )

    return {
        "grupos_avaliados": grupos_avaliados,
        "grupos_acima_080": grupos_acima_080,
        "cobertura_080": cobertura_080,
        "imposturas_total": imposturas_total,
        "erro_baixo": erro_baixo,
        "erro_alto": erro_alto,
        "taxa_impostura": taxa_impostura,
        "acertos_total": acertos_total,
        "acerto_baixo": acerto_baixo,
        "acerto_alto": acerto_alto,
    }


def analisar_imposturas(previsoes):
    """
    Analisa as imposturas identificadas no período.

    Retorna:
        - quantidade de imposturas por membro;
        - quantidade de vezes que cada identidade foi imitada;
        - frequência de cada par impostor -> identidade imitada.
    """

    imposturas = [
        previsao
        for previsao in previsoes
        if previsao["categoria"] in ("ERRO_BAIXO", "ERRO_ALTO")
        and previsao["probabilidade"] >= 0.80
    ]

    por_impostor = Counter(
        previsao["autor_real"]
        for previsao in imposturas
    )

    identidades_imitadas = Counter(
        previsao["autor_predito"]
        for previsao in imposturas
    )

    pares = Counter(
        (
            previsao["autor_real"],
            previsao["autor_predito"],
        )
        for previsao in imposturas
    )

    return {
        "imposturas": imposturas,
        "por_impostor": dict(por_impostor),
        "identidades_imitadas": dict(identidades_imitadas),
        "pares_impostura": {
            f"{impostor} → {imitado}": quantidade
            for (impostor, imitado), quantidade in pares.items()
        },
    }

def detalhar_imposturas(previsoes):
    """
    Retorna as ocorrências individuais de impostura
    identificadas no período.
    """

    return [
        {
            "autor_real": previsao["autor_real"],
            "autor_predito": previsao["autor_predito"],
            "probabilidade": previsao["probabilidade"],
            "categoria": previsao["categoria"],
            "data_hora": previsao["processado_em"],
            "mensagem": previsao["mensagem"],
        }
        for previsao in previsoes
        if previsao["categoria"] in ("ERRO_BAIXO", "ERRO_ALTO")
        and previsao["probabilidade"] >= 0.80
    ]

def montar_dados_diario(previsoes):
    """
    Monta o pacote objetivo de dados que será usado
    posteriormente pelo gerador do Diário do Inspetor.
    """

    kpis = calcular_kpis(previsoes)
    analise_imposturas = analisar_imposturas(previsoes)
    detalhes_imposturas = detalhar_imposturas(previsoes)

    analise_imposturas.pop("imposturas")

    return {
        "kpis": kpis,
        "imposturas": analise_imposturas,
        "detalhes_imposturas": detalhes_imposturas,
    }