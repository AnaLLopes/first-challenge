
import requests
import time
import json

from codecarbon import EmissionsTracker

MODEL = "llama3.2:1b"

PROMPT = """
Explique o que é inteligência artificial sustentável e apresente
cinco práticas que desenvolvedores podem adotar para reduzir o
consumo de energia, água e as emissões de carbono ao utilizar
modelos de linguagem (LLMs).

Para cada prática, explique brevemente como ela contribui para
a sustentabilidade.

Finalize com um exemplo prático de aplicação em um projeto
de software.

Utilize linguagem clara, objetiva e técnica,
com aproximadamente 300 palavras.
"""

# Monitorar a máquina inteira, pois o Ollama executa
# o modelo em outro processo.
tracker = EmissionsTracker(
    project_name="teste_llm_local",
    tracking_mode="machine",
    output_file="codecarbon.csv",
    log_level="error",
    measure_power_secs=1
)

print("Iniciando experimento...")

tracker.start()
inicio = time.perf_counter()

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=600
    )

    response.raise_for_status()
    resultado = response.json()

finally:
    duracao = time.perf_counter() - inicio
    tracker.stop()

# Dados da resposta
print("\n======= RESPOSTA DA IA =======")
print(resultado["response"])

print("\n======= MÉTRICAS DO OLLAMA =======")
print("Modelo:", resultado["model"])
print("Tokens de entrada:", resultado.get("prompt_eval_count"))
print("Tokens de saída:", resultado.get("eval_count"))
print("Duração total (s):", resultado.get("total_duration", 0) / 1e9)
print("Duração Python (s):", round(duracao, 2))

# Dados do CodeCarbon
dados = tracker.final_emissions_data

print("\n======= CODECARBON =======")
print("Energia (kWh):", dados.energy_consumed)
print("Carbono (kg CO2eq):", tracker.final_emissions)
print("Duração monitorada (s):", dados.duration)

# Salvar resultado
dados_teste = {
    "modelo": MODEL,
    "tokens_entrada": resultado.get("prompt_eval_count"),
    "tokens_saida": resultado.get("eval_count"),
    "tempo_resposta_segundos": duracao,
    "energia_kwh": dados.energy_consumed,
    "carbono_kg": tracker.final_emissions
}

with open("resultado_teste.json", "w", encoding="utf-8") as arquivo:
    json.dump(dados_teste, arquivo, indent=4, ensure_ascii=False)

print("\nArquivo salvo: resultado_teste.json")
print("Arquivo salvo: codecarbon.csv")
