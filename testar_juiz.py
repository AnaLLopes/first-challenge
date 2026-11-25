#!/usr/bin/env python3
"""TokenWise - descobre quais modelos da Mistral o seu plano libera para o juiz.

Faz UMA chamada curta a cada modelo candidato e mostra o resultado:
    OK      -> o modelo funciona no seu plano
    403     -> o modelo não está liberado no seu plano
    429     -> limite de uso atingido (tente de novo em alguns minutos)

Uso:
    python testar_juiz.py
"""

import os
import time

from dotenv import load_dotenv

from coletar_aberto import chamar_mistral

# Em ordem de preferência para juiz (do mais forte para o mais simples)
CANDIDATOS = [
    "mistral-large-2512",
    "mistral-large-4",
    "glm-5-2",
    "zai-glm-5-3",
    "mistral-medium-latest",
    "mistral-small-2603",
    "deepseek-v4-1-flash-1017",
]

load_dotenv()
chave = os.getenv("MISTRAL_API_KEY")
if not chave:
    raise SystemExit("Erro: defina MISTRAL_API_KEY no arquivo .env.")

print("Testando os modelos (uma chamada curta para cada)...\n")
funcionam = []
for modelo in CANDIDATOS:
    try:
        resposta = chamar_mistral(chave, modelo,
                                  'Responda somente com este JSON: {"ok": true}', timeout=60)
        texto = resposta["choices"][0]["message"]["content"].strip()
        print(f"  OK   {modelo:<28} resposta: {texto[:40]}")
        funcionam.append(modelo)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        codigo = "403" if "403" in msg else "429" if "429" in msg else "ERRO"
        print(f"  {codigo:<4} {modelo:<28} {msg[:90]}")
    time.sleep(3)

print()
if funcionam:
    print("Modelos que funcionam no seu plano:", ", ".join(funcionam))
    print(f"Sugestão para o .env:  MODELO_JUIZ={funcionam[0]}")
else:
    print("Nenhum modelo respondeu. Confira no painel da Mistral se a conta está verificada.")