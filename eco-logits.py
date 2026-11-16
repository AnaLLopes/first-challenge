
import json
import time
import litellm

from ecologits import EcoLogits

# Mesmo modelo e prompt do experimento anterior
MODEL = "ollama/llama3.2:1b"

PROMPT = """
Explique o que é inteligência artificial sustentável e apresente
cinco práticas que desenvolvedores podem adotar para reduzir
o consumo de energia, água e as emissões de carbono ao utilizar
modelos de linguagem (LLMs).

Para cada prática, explique brevemente como ela contribui
para a sustentabilidade.

Finalize com um exemplo prático de aplicação em um projeto
de software.

Utilize linguagem clara, objetiva e técnica,
com aproximadamente 300 palavras.
"""

# Inicializa o EcoLogits para integração com LiteLLM
EcoLogits.init(providers=["litellm"])

print("Enviando prompt para o Ollama...")

inicio = time.perf_counter()

try:
    response = litellm.completion(
        model=MODEL,
        api_base="http://localhost:11434",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        temperature=0.2,
        max_tokens=450
    )

    tempo = time.perf_counter() - inicio

    # Resposta do modelo
    print("\n======= RESPOSTA =======")
    print(response.choices[0].message.content)

    # Dados da geração
    print("\n======= DADOS DA GERAÇÃO =======")
    print("Modelo:", response.model)
    print("Tokens de entrada:", response.usage.prompt_tokens)
    print("Tokens de saída:", response.usage.completion_tokens)
    print("Tempo da chamada:", round(tempo, 2), "segundos")

    # Dados do EcoLogits
    print("\n======= ECOLOGITS =======")

    impactos = getattr(response, "impacts", None)

    if impactos is None:
        print("EcoLogits não anexou dados de impacto.")
    else:
        print("Energia:", impactos.energy)
        print("Carbono:", impactos.gwp)
        print("Água (WCF):", impactos.wcf)

        print("\nAvisos:", impactos.warnings)
        print("Erros:", impactos.errors)

        with open(
            "resultado_ecologits.json",
            "w",
            encoding="utf-8"
        ) as arquivo:
            json.dump(
                impactos.model_dump(mode="json"),
                arquivo,
                indent=4,
                ensure_ascii=False
            )

        print("\nResultado salvo em resultado_ecologits.json")

except Exception as erro:
    print("\nFalha na execução:")
    print(type(erro).__name__, erro)
