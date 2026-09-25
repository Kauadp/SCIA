import os

from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "A variável de ambiente GROQ_API_KEY não foi encontrada."
    )


client = Groq(
    api_key=GROQ_API_KEY
)


MODELO = "openai/gpt-oss-20b"


def gerar_resposta(
    system_prompt: str,
    user_prompt: str,
    temperatura: float = 0.8,
    max_tokens: int = 100,
) -> str:

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=temperatura,
        max_completion_tokens=max_tokens,
        reasoning_effort="low",
        include_reasoning=False,
    )
    
    print("\n===== DEBUG GROQ =====")
    print("Modelo:", resposta.model)
    print("Reasoning:", resposta.choices[0].message.reasoning)
    print("Content:", repr(resposta.choices[0].message.content))
    print("======================\n")

    return resposta.choices[0].message.content.strip()



