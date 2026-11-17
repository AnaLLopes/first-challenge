"""Executa prompts no Gemini e salva os dados de geração e impactos EcoLogits."""

import json
import os
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ecologits import EcoLogits
from google import genai
from langdetect import DetectorFactory, LangDetectException, detect_langs


PASTA = Path(__file__).resolve().parent
load_dotenv(PASTA / ".env")

ARQUIVO_ENTRADA = PASTA / "prompts_gemini.json"
ARQUIVO_SAIDA = PASTA / "dataset_gemini.json"
MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
PAUSA_ENTRE_PROMPTS_SEGUNDOS = float(os.getenv("SLEEP_ENTRE_CHAMADAS", "2"))

NOMES_IDIOMAS = {
    "de": "Alemão",
    "en": "Inglês",
    "es": "Espanhol",
    "fr": "Francês",
    "it": "Italiano",
    "pt": "Português",
}

INDICADORES = ("energy", "gwp", "adpe", "pe", "wcf")

DetectorFactory.seed = 0


def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(
        caractere for caractere in texto
        if not unicodedata.combining(caractere)
    )


def classificar_prompt(prompt: str) -> dict[str, Any]:
    texto = normalizar_texto(prompt)
    categorias = {
        "geracao_codigo": [
            "crie um codigo", "escreva um codigo",
            "implemente", "desenvolva uma funcao",
            "crie uma funcao", "programar",
            "escreva uma classe", "gere um script",
        ],
        "depuracao_codigo": [
            "corrija o erro", "debug", "depure",
            "identifique o erro", "corrija este codigo",
            "por que este codigo", "erro no codigo",
        ],
        "calculo_matematico": [
            "quanto e", "calcule", "resolva a equacao",
            "qual o resultado", "multiplique",
            "divida", "some", "subtraia",
            "raiz quadrada", "porcentagem",
        ],
        "analise_dados": [
            "analise os dados", "analise esta planilha",
            "identifique tendencias", "estatistica",
            "analise os resultados", "compare os dados",
        ],
        "resumo": [
            "resuma", "faca um resumo",
            "sintetize", "resuma este texto",
        ],
        "traducao": [
            "traduza", "traduzir",
            "traduza para", "versao em ingles",
        ],
        "geracao_texto": [
            "escreva um texto", "redija",
            "escreva um email", "crie um artigo",
            "elabore um texto", "escreva uma redacao",
        ],
        "criatividade": [
            "crie uma historia", "invente",
            "crie um poema", "crie um personagem",
            "ideia criativa", "escreva uma ficcao",
        ],
        "planejamento": [
            "monte um cronograma", "crie um plano",
            "organize uma rotina", "planeje",
            "monte um planejamento",
        ],
        "classificacao": [
            "classifique", "categorize",
            "separe por categoria", "identifique o sentimento",
            "classificacao de",
        ],
        "extracao_dados": [
            "extraia", "extraia os dados",
            "identifique os nomes", "extraia informacoes",
            "leia este documento e extraia",
        ],
    }

    pontuacoes = {
        categoria: pontuacao
        for categoria, palavras_chave in categorias.items()
        if (pontuacao := sum(
            1 for palavra in palavras_chave
            if normalizar_texto(palavra) in texto
        )) > 0
    }

    encontradas = sorted(
        pontuacoes.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if encontradas:
        principal = encontradas[0][0]
        secundarias = [categoria for categoria, _ in encontradas[1:]]
    elif re.search(
        r"\b(quem|qual|quais|quando|onde|como|por que|o que)\b",
        texto,
    ):
        principal = "pergunta_informativa"
        secundarias = []
    else:
        principal = "outro"
        secundarias = []

    return {
        "categoria_principal": principal,
        "categorias_secundarias": secundarias,
        "pontuacoes": pontuacoes,
    }


def carregar_prompts(caminho: Path) -> list[dict[str, str]]:
    """Lê uma lista JSON contendo somente textos de prompts."""
    with caminho.open(encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    if not isinstance(dados, list) or not dados:
        raise ValueError("O arquivo de entrada deve conter uma lista JSON não vazia.")

    prompts: list[dict[str, str]] = []
    for indice, item in enumerate(dados, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"O prompt do item {indice} não pode ser vazio.")

        prompts.append({
            "prompt_id": f"P{indice:03d}",
            "prompt": item.strip(),
        })

    return prompts


def carregar_dataset(caminho: Path) -> list[dict[str, Any]]:
    if not caminho.exists():
        return []

    with caminho.open(encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    if not isinstance(dados, list):
        raise ValueError(f"O dataset existente não é uma lista JSON: {caminho}")
    return dados


def salvar_dataset(caminho: Path, registros: list[dict[str, Any]]) -> None:
    """Substitui o JSON atomicamente para preservar o dataset anterior."""
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    with temporario.open("w", encoding="utf-8") as arquivo:
        json.dump(registros, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    temporario.replace(caminho)


def analisar_prompt(prompt: str) -> dict[str, Any]:
    texto = unicodedata.normalize("NFC", prompt)
    palavras = re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", texto, flags=re.UNICODE)

    idioma: dict[str, Any]
    try:
        probabilidades = detect_langs(texto)
        principal = probabilidades[0]
        idioma = {
            "codigo": principal.lang,
            "nome": NOMES_IDIOMAS.get(principal.lang, principal.lang),
            "probabilidade": round(principal.prob, 4),
            "alternativas": [
                {"codigo": item.lang, "probabilidade": round(item.prob, 4)}
                for item in probabilidades
            ],
            "status": "identificado",
        }
    except LangDetectException as erro:
        idioma = {
            "codigo": None,
            "nome": "não identificado",
            "probabilidade": None,
            "alternativas": [],
            "status": "não identificado",
            "erro": str(erro),
        }

    return {
        "quantidade_palavras": len(palavras),
        "quantidade_caracteres": len(texto),
        "idioma": idioma,
        "contagem_caracteres": (
            "Inclui espaços, pontuação e quebras de linha; "
            "texto normalizado em Unicode NFC."
        ),
    }


def extrair_valor(valor: Any) -> Any:
    if valor is None or isinstance(valor, (str, int, float, bool)):
        return valor

    minimo = getattr(valor, "min", None)
    maximo = getattr(valor, "max", None)
    if minimo is not None and maximo is not None:
        return {
            "min": minimo,
            "mean": getattr(valor, "mean", (minimo + maximo) / 2),
            "max": maximo,
        }

    return str(valor)


def extrair_indicador(indicador: Any) -> dict[str, Any] | None:
    if indicador is None:
        return None

    return {
        "nome": getattr(indicador, "name", None),
        "valor": extrair_valor(getattr(indicador, "value", None)),
        "unidade": getattr(indicador, "unit", None),
    }


def extrair_fase(fase: Any) -> dict[str, Any] | None:
    if fase is None:
        return None
    return {
        nome: extrair_indicador(getattr(fase, nome, None))
        for nome in INDICADORES
    }


def extrair_impactos(response: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    impactos = getattr(response, "impacts", None)
    if impactos is None:
        return None, {
            "disponivel": False,
            "avisos": [],
            "erros": [],
            "observacao": (
                "EcoLogits não anexou impactos. Valores ausentes não "
                "devem ser interpretados como zero."
            ),
        }

    dados = {
        "total": extrair_fase(impactos),
        "uso": extrair_fase(getattr(impactos, "usage", None)),
        "incorporados": extrair_fase(getattr(impactos, "embodied", None)),
    }
    diagnostico = {
        "disponivel": True,
        "avisos": [
            str(item) for item in (getattr(impactos, "warnings", None) or [])
        ],
        "erros": [
            str(item) for item in (getattr(impactos, "errors", None) or [])
        ],
        "observacao": (
            "Os indicadores são estimativas modeladas. O total já inclui "
            "o impacto total; não some total, uso e incorporados."
        ),
    }
    return dados, diagnostico


def processar_prompt(
    item: dict[str, str],
    indice: int,
    total: int,
    cliente: Any,
    run_id: str,
) -> dict[str, Any]:
    inicio = time.perf_counter()
    registro: dict[str, Any] = {
        "registro_id": str(uuid.uuid4()),
        "run_id": run_id,
        "prompt_id": item["prompt_id"],
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
        "provedor": "Google Gemini",
        "modelo": MODELO,
        "classificacao": classificar_prompt(item["prompt"]),
        "prompt": item["prompt"],
        "analise_prompt": analisar_prompt(item["prompt"]),
        "status": "iniciado",
        "duracao_segundos": None,
        "tokens": {"entrada": None, "saida": None, "total": None},
        "resposta_llm": None,
        "impactos_ambientais": None,
        "diagnostico_ecologits": None,
        "erro_execucao": None,
    }

    print(f"[{indice}/{total}] Processando {item['prompt_id']}...")
    try:
        response = cliente.models.generate_content(
            model=MODELO,
            contents=item["prompt"],
        )
        registro["duracao_segundos"] = round(time.perf_counter() - inicio, 6)
        registro["resposta_llm"] = getattr(response, "text", None)

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            registro["tokens"] = {
                "entrada": getattr(usage, "prompt_token_count", None),
                "saida": getattr(usage, "candidates_token_count", None),
                "total": getattr(usage, "total_token_count", None),
            }

        impactos, diagnostico = extrair_impactos(response)
        registro["impactos_ambientais"] = impactos
        registro["diagnostico_ecologits"] = diagnostico
        registro["status"] = (
            "sucesso" if impactos is not None else "sucesso_sem_dados_ambientais"
        )
    except Exception as erro:
        registro["duracao_segundos"] = round(time.perf_counter() - inicio, 6)
        registro["status"] = "erro"
        registro["erro_execucao"] = {
            "tipo": type(erro).__name__,
            "mensagem": str(erro),
        }
        print(f"Erro em {item['prompt_id']}: {type(erro).__name__}: {erro}")

    return registro


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "cole_sua_chave_aqui":
        raise RuntimeError(
            "GEMINI_API_KEY não definida. Configure-a no arquivo .env "
            "ou no ambiente antes de executar."
        )
    if PAUSA_ENTRE_PROMPTS_SEGUNDOS < 0:
        raise ValueError("SLEEP_ENTRE_CHAMADAS não pode ser negativo.")
    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de prompts não encontrado: {ARQUIVO_ENTRADA}")

    prompts = carregar_prompts(ARQUIVO_ENTRADA)
    dataset = carregar_dataset(ARQUIVO_SAIDA)

    EcoLogits.init(providers=["google_genai"])
    cliente = genai.Client(api_key=api_key)
    run_id = str(uuid.uuid4())

    print(f"Prompts: {len(prompts)} | Modelo: {MODELO}")
    for indice, item in enumerate(prompts, start=1):
        registro = processar_prompt(item, indice, len(prompts), cliente, run_id)
        dataset.append(registro)
        salvar_dataset(ARQUIVO_SAIDA, dataset)
        print(
            f"Status: {registro['status']} | "
            f"Duração: {registro['duracao_segundos']} s"
        )

        if indice < len(prompts):
            time.sleep(PAUSA_ENTRE_PROMPTS_SEGUNDOS)

    print(f"Dataset atualizado: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()
