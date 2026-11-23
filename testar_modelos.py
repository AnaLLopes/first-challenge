"""
Passo 1 do piloto TokenWise: confirma que a chave do Gemini funciona
nos dois modelos (A = econômica, B = referência) e mostra os tokens usados.

Uso:
    pip install google-genai python-dotenv
    python testar_modelos.py

A chave é lida do arquivo .env (GEMINI_API_KEY=...), o mesmo do coletor.
"""

import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

CONFIGURACOES = {
    "A (econômica)": "gemini-3.5-flash-lite",
    "B (referência)": "gemini-3.6-flash",
}

# Mesma instrução e mesma correção usadas no coletar_piloto.py
PERGUNTA_TESTE = (
    "Resolva a questão de múltipla escolha abaixo. "
    "Ao final, escreva em uma linha separada exatamente no formato "
    "'Resposta: X', em que X é a letra da alternativa correta.\n\n"
    "Quanto é 15% de 200?\n"
    "A) 15\nB) 20\nC) 30\nD) 35\nE) 40"
)
GABARITO = "C"
PADRAO_RESPOSTA = re.compile(
    r"resposta\W{0,4}:\W{0,4}([A-E])(?![A-Za-zÀ-ÿ])",
    re.IGNORECASE,
)


def extrair_letra(texto):
    achados = PADRAO_RESPOSTA.findall(texto or "")
    return achados[-1].upper() if achados else None


def main():
    chave = os.getenv("GEMINI_API_KEY")
    if not chave:
        print("ERRO: GEMINI_API_KEY não encontrada. Confira o arquivo .env.")
        return

    cliente = genai.Client(api_key=chave)

    for nome, modelo in CONFIGURACOES.items():
        print(f"\n=== Configuração {nome}: {modelo} ===")
        try:
            resposta = cliente.models.generate_content(
                model=modelo,
                contents=PERGUNTA_TESTE,
            )
        except Exception as erro:
            print(f"FALHOU: {erro}")
            continue

        texto = (resposta.text or "").strip()
        uso = resposta.usage_metadata
        entrada = uso.prompt_token_count or 0
        saida = uso.candidates_token_count or 0
        raciocinio = getattr(uso, "thoughts_token_count", None) or 0

        letra = extrair_letra(texto)
        if letra is None:
            situacao = "SEM LETRA (não achou a linha 'Resposta: X')"
        else:
            situacao = "ACERTOU" if letra == GABARITO else "ERROU"
        print(f"Letra lida: {letra}  ->  {situacao}")
        print(f"Tokens de entrada:    {entrada}")
        print(f"Tokens de saída:      {saida}")
        print(f"Tokens de raciocínio: {raciocinio}  (cobrados como saída)")


if __name__ == "__main__":
    main()