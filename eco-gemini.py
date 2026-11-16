"""Run a Gemini generation request with EcoLogits impact tracking."""

import os
import json
from google import genai
from ecologits import EcoLogits

# 1. Inicializa o EcoLogits para o provedor Google Gemini
EcoLogits.init(providers=["google_genai"])

# 2. Configura o cliente Gemini
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODELO = "gemini-3.5-flash-lite"

PROMPT = """
Explique o que é inteligência artificial sustentável
e apresente cinco práticas para reduzir o consumo
de energia, água e emissões de carbono no uso de LLMs.
"""

# 3. Executa a requisição
response = client.models.generate_content(
    model=MODELO,
    contents=PROMPT
)

# 4. Captura os impactos ambientais
impacts = getattr(response, "impacts", None)

def extrair_valor(valor):
    """Converte valores simples ou intervalos em JSON."""
    if valor is None:
        return None

    if hasattr(valor, "min") and hasattr(valor, "max"):
        return {
            "min": valor.min,
            "max": valor.max
        }

    return valor

resultado = {
    "modelo": MODELO,
    "provedor": "Google Gemini",
    "prompt": PROMPT,
    "resposta": response.text,
    "tokens": {
        "entrada": getattr(
            response.usage_metadata,
            "prompt_token_count",
            None
        ),
        "saida": getattr(
            response.usage_metadata,
            "candidates_token_count",
            None
        ),
        "total": getattr(
            response.usage_metadata,
            "total_token_count",
            None
        )
    },
    "impactos_ambientais": None
}

if impacts:
    wcf = getattr(impacts, "wcf", None)
    energia = getattr(impacts, "energy", None)
    carbono = getattr(impacts, "gwp", None)

    resultado["impactos_ambientais"] = {
        "agua": {
            "valor": extrair_valor(wcf.value) if wcf else None,
            "unidade": wcf.unit if wcf else "L"
        },
        "energia": {
            "valor": extrair_valor(energia.value) if energia else None,
            "unidade": energia.unit if energia else "kWh"
        },
        "carbono": {
            "valor": extrair_valor(carbono.value) if carbono else None,
            "unidade": carbono.unit if carbono else "kgCO2eq"
        }
    }

    resultado["avisos"] = [
        str(aviso) for aviso in (impacts.warnings or [])
    ]
else:
    resultado["erro"] = (
        "EcoLogits não retornou impactos para esta execução."
    )

print(json.dumps(resultado, indent=4, ensure_ascii=False))
 