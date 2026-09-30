from .cliente import gerar_resposta
from .prompts import gerar_prompt


def gerar_mensagem(
    categoria: str,
    autor_real: str,
    autor_predito: str,
    personalidade: str | None = None,
) -> tuple[str, str]:

    prompt = gerar_prompt(
        categoria=categoria,
        autor_real=autor_real,
        autor_predito=autor_predito,
        personalidade=personalidade,
    )

    texto = gerar_resposta(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        temperatura=prompt["temperatura"],
        max_tokens=150,
    )

    return texto, prompt["personalidade"]