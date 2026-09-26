from .cliente import gerar_resposta
from .prompts import gerar_prompt


def gerar_mensagem(
    categoria: str,
    personalidade: str | None = None,
) -> tuple[str, str]:

    prompt = gerar_prompt(
        categoria=categoria,
        personalidade=personalidade,
    )

    texto = gerar_resposta(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        temperatura=prompt["temperatura"],
        max_tokens=150,
    )

    return texto, prompt["personalidade"]