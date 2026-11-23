"""Executa prompts no Gemini e salva os dados de geração e impactos EcoLogits."""

import argparse
import csv
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
from prompt_complexity import analisar_complexidade_prompt


PASTA = Path(__file__).resolve().parent
load_dotenv(PASTA / ".env")

ARQUIVO_ENTRADA = PASTA / "meus_prompts_traduzidos.csv"
ARQUIVO_SAIDA = PASTA / "dataset_gemini.json"
ARQUIVO_CSV = PASTA / "dataset_gemini.csv"
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
    """Lê o CSV de prompts e preserva id, tarefa e as duas versões."""
    colunas_obrigatorias = [
        "id", "tarefa", "prompt_original", "prompt_otimizado",
    ]

    with caminho.open("r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        if not leitor.fieldnames:
            raise ValueError("O arquivo CSV não possui cabeçalho.")

        ausentes = [
            coluna for coluna in colunas_obrigatorias
            if coluna not in leitor.fieldnames
        ]
        if ausentes:
            raise ValueError(
                "Colunas obrigatórias ausentes no CSV: " + ", ".join(ausentes)
            )

        dados = list(leitor)

    if not dados:
        raise ValueError("O arquivo CSV não contém prompts.")

    prompts: list[dict[str, str]] = []
    ids_vistos: set[str] = set()

    for indice, item in enumerate(dados, start=1):
        prompt_id = str(item["id"] or "").strip()
        tarefa = str(item["tarefa"] or "").strip()
        prompt_original = str(item["prompt_original"] or "").strip()
        prompt_otimizado = str(item["prompt_otimizado"] or "").strip()

        if not prompt_id:
            raise ValueError(f"O item {indice} não possui id.")
        if prompt_id in ids_vistos:
            raise ValueError(f"id duplicado no CSV: {prompt_id}")
        if not prompt_original:
            raise ValueError(f"prompt_original vazio no id {prompt_id}.")
        if not prompt_otimizado:
            raise ValueError(f"prompt_otimizado vazio no id {prompt_id}.")

        ids_vistos.add(prompt_id)
        prompts.append({
            "prompt_id": prompt_id,
            "tarefa": tarefa,
            "prompt_original": prompt_original,
            "prompt_otimizado": prompt_otimizado,
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


PRECOS_USD_POR_MILHAO = {
    "gemini-3.5-flash-lite": {"entrada": 0.30, "saida": 2.50},
    "gemini-3.6-flash": {"entrada": 0.75, "saida": 3.75},
}


def extrair_metricas_uso(response: Any) -> dict[str, Any]:
    """Extrai métricas públicas do uso sem depender de uma única versão do SDK."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {"entrada": 0, "saida_visivel": 0, "raciocinio": 0,
                "cache": 0, "saida_total_cobrada": 0, "total": 0,
                "detalhes": {}}

    entrada = getattr(usage, "prompt_token_count", None) or 0
    saida_visivel = getattr(usage, "candidates_token_count", None) or 0
    raciocinio = getattr(usage, "thoughts_token_count", None) or 0
    cache = getattr(usage, "cached_content_token_count", None) or 0
    total = getattr(usage, "total_token_count", None) or (
        entrada + saida_visivel + raciocinio
    )

    detalhes: dict[str, Any] = {}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            detalhes = model_dump(mode="json", exclude_none=True)
        except Exception:
            detalhes = {}
    if not detalhes:
        for nome in dir(usage):
            if nome.startswith("_"):
                continue
            try:
                valor = getattr(usage, nome)
            except Exception:
                continue
            if isinstance(valor, (str, int, float, bool, list, dict, type(None))):
                detalhes[nome] = valor

    return {
        "entrada": entrada,
        "saida_visivel": saida_visivel,
        "raciocinio": raciocinio,
        "cache": cache,
        "saida_total_cobrada": saida_visivel + raciocinio,
        "total": total,
        "detalhes": detalhes,
    }


def calcular_custo_equivalente(modelo: str, entrada: int, saida_total: int) -> dict[str, Any]:
    """Estima o custo pago por milhão de tokens; não é cobrança da chave gratuita."""
    precos = PRECOS_USD_POR_MILHAO.get(modelo)
    if precos is None:
        return {
            "preco_entrada_usd_milhao": None,
            "preco_saida_usd_milhao": None,
            "custo_equivalente_usd": None,
            "observacao": "Modelo sem tabela de preço configurada.",
        }

    custo = (
        entrada / 1_000_000 * precos["entrada"]
        + saida_total / 1_000_000 * precos["saida"]
    )
    return {
        "preco_entrada_usd_milhao": precos["entrada"],
        "preco_saida_usd_milhao": precos["saida"],
        "custo_equivalente_usd": round(custo, 12),
        "observacao": "Estimativa equivalente ao plano pago; não representa cobrança da conta gratuita.",
    }


def achatavel(valor: Any, prefixo: str = "", destino: dict[str, str] | None = None) -> dict[str, str]:
    """Transforma campos aninhados em colunas CSV com nomes únicos."""
    destino = destino if destino is not None else {}
    if isinstance(valor, dict):
        for chave, filho in valor.items():
            nome = f"{prefixo}.{chave}" if prefixo else str(chave)
            achatavel(filho, nome, destino)
    elif isinstance(valor, list):
        destino[prefixo or "lista"] = json.dumps(valor, ensure_ascii=False)
    elif valor is None:
        destino[prefixo or "valor"] = ""
    else:
        destino[prefixo or "valor"] = str(valor)
    return destino


def exportar_csv(caminho: Path, registros: list[dict[str, Any]]) -> None:
    """Regrava o CSV com uma linha por registro e cabeçalho da união das colunas."""
    registros_exportacao: list[dict[str, Any]] = []
    for registro in registros:
        registro_exportacao = dict(registro)
        tokens_exportacao = dict(registro_exportacao.get("tokens") or {})
        entrada = tokens_exportacao.get("entrada", 0) or 0
        saida = tokens_exportacao.get("saida", 0) or 0
        raciocinio = tokens_exportacao.get("raciocinio", 0) or 0
        tokens_exportacao.setdefault("raciocinio", raciocinio)
        tokens_exportacao.setdefault("cache", 0)
        tokens_exportacao.setdefault("saida_total_cobrada", saida + raciocinio)
        registro_exportacao["tokens"] = tokens_exportacao

        if not registro_exportacao.get("metricas_gemini"):
            registro_exportacao["metricas_gemini"] = {
                "custo": calcular_custo_equivalente(
                    registro_exportacao.get("modelo", ""),
                    entrada,
                    saida + raciocinio,
                ),
                "modelo_resposta": None,
                "finish_reason": None,
                "response_id": None,
                "usage_metadata": {},
            }
        registros_exportacao.append(registro_exportacao)

    linhas = [achatavel(registro) for registro in registros_exportacao]
    colunas = sorted({chave for linha in linhas for chave in linha})

    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    with temporario.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=colunas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(linhas)
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
    versao_prompt: str,
) -> dict[str, Any]:
    prompt = item[f"prompt_{versao_prompt}"]
    inicio = time.perf_counter()
    try:
        complexidade_prompt = analisar_complexidade_prompt(prompt)
    except Exception as erro:
        complexidade_prompt = {
            "pontuacao": None,
            "nivel": None,
            "metodo": "heuristico",
            "metricas": {},
            "indicadores": {},
            "status_analise": "erro",
            "erro": {
                "tipo": type(erro).__name__,
                "mensagem": str(erro),
            },
        }

    registro: dict[str, Any] = {
        "registro_id": str(uuid.uuid4()),
        "run_id": run_id,
        "prompt_id": item["prompt_id"],
        "tarefa": item["tarefa"],
        "versao_prompt": versao_prompt,
        "prompt_original": item["prompt_original"],
        "prompt_otimizado": item["prompt_otimizado"],
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
        "provedor": "Google Gemini",
        "modelo": MODELO,
        "classificacao": classificar_prompt(prompt),
        "prompt": prompt,
        "analise_prompt": analisar_prompt(prompt),
        "complexidade_prompt": complexidade_prompt,
        "metricas_gemini": {
            "response_id": None,
            "finish_reason": None,
            "modelo_resposta": None,
            "usage_metadata": {},
        },
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
            contents=prompt,
        )
        registro["duracao_segundos"] = round(time.perf_counter() - inicio, 6)
        registro["resposta_llm"] = getattr(response, "text", None)

        metricas_uso = extrair_metricas_uso(response)
        registro["tokens"] = {
            "entrada": metricas_uso["entrada"],
            "saida": metricas_uso["saida_visivel"],
            "raciocinio": metricas_uso["raciocinio"],
            "cache": metricas_uso["cache"],
            "saida_total_cobrada": metricas_uso["saida_total_cobrada"],
            "total": metricas_uso["total"],
        }

        custo = calcular_custo_equivalente(
            MODELO,
            metricas_uso["entrada"],
            metricas_uso["saida_total_cobrada"],
        )
        finish_reason = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)

        registro["metricas_gemini"] = {
            "response_id": getattr(response, "response_id", None),
            "finish_reason": str(finish_reason) if finish_reason is not None else None,
            "modelo_resposta": getattr(response, "model_version", None),
            "usage_metadata": metricas_uso["detalhes"],
            "custo": custo,
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
    parser = argparse.ArgumentParser(
        description="Coleta prompts originais e otimizados no Gemini."
    )
    parser.add_argument("--entrada", default=str(ARQUIVO_ENTRADA))
    parser.add_argument("--saida-json", default=str(ARQUIVO_SAIDA))
    parser.add_argument("--saida-csv", default=str(ARQUIVO_CSV))
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument(
        "--versoes",
        choices=("original", "otimizado", "ambos"),
        default="ambos",
    )
    args = parser.parse_args()

    caminho_entrada = Path(args.entrada)
    caminho_json = Path(args.saida_json)
    caminho_csv = Path(args.saida_csv)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "cole_sua_chave_aqui":
        raise RuntimeError(
            "GEMINI_API_KEY não definida. Configure-a no arquivo .env "
            "ou no ambiente antes de executar."
        )
    if PAUSA_ENTRE_PROMPTS_SEGUNDOS < 0:
        raise ValueError("SLEEP_ENTRE_CHAMADAS não pode ser negativo.")
    if not caminho_entrada.exists():
        raise FileNotFoundError(
            f"Arquivo de prompts não encontrado: {caminho_entrada}"
        )

    prompts = carregar_prompts(caminho_entrada)
    dataset = carregar_dataset(caminho_json)
    exportar_csv(caminho_csv, dataset)

    EcoLogits.init(providers=["google_genai"])
    cliente = genai.Client(api_key=api_key)
    run_id = str(uuid.uuid4())

    if args.limite is not None:
        if args.limite < 1:
            raise ValueError("--limite deve ser pelo menos 1.")
        prompts = prompts[:args.limite]

    versoes = (
        ("original", "otimizado")
        if args.versoes == "ambos"
        else (args.versoes,)
    )

    ids_processados = {
        (str(registro.get("prompt_id")), registro.get("versao_prompt"))
        for registro in dataset
        if registro.get("versao_prompt")
    }
    pendentes: list[tuple[dict[str, str], str]] = []
    for item in prompts:
        for versao in versoes:
            chave = (str(item["prompt_id"]), versao)
            if chave not in ids_processados:
                pendentes.append((item, versao))

    total = len(pendentes)
    print(
        f"Linhas CSV: {len(prompts)} | Versões: {versoes} | "
        f"Já registradas: {len(ids_processados)} | Pendentes: {total} | "
        f"Modelo: {MODELO}"
    )

    for indice, (item, versao) in enumerate(pendentes, start=1):
        print(f"[{indice}/{total}] id={item['prompt_id']} versão={versao}")
        registro = processar_prompt(
            item,
            indice,
            total,
            cliente,
            run_id,
            versao,
        )
        dataset.append(registro)
        salvar_dataset(caminho_json, dataset)
        exportar_csv(caminho_csv, dataset)
        print(
            f"Status: {registro['status']} | "
            f"Duração: {registro['duracao_segundos']} s"
        )

        if indice < total:
            time.sleep(PAUSA_ENTRE_PROMPTS_SEGUNDOS)

    print(f"Dataset JSON atualizado: {caminho_json}")
    print(f"Dataset CSV atualizado: {caminho_csv}")


if __name__ == "__main__":
    main()
