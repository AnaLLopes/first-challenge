#!/usr/bin/env python3
"""TokenWise - coletor do piloto (configuração A x configuração B).

Envia a MESMA questão (com gabarito) às duas configurações, confere se cada
uma acertou, calcula o custo equivalente e grava o rótulo usado no
treinamento do recomendador. Uma linha por questão no CSV de saída.

    A = configuração econômica   (gemini-3.5-flash-lite)
    B = configuração de referência (gemini-3.6-flash)

Uso:
    python coletar_piloto.py --entrada questoes_charles.csv --executor charles

Entrada (CSV): id, area, fonte, enunciado, gabarito
    - enunciado: texto completo da questão, já com as alternativas A) a E)
    - gabarito:  letra correta (A, B, C, D ou E)

Retomada: cada questão é gravada assim que A e B terminam. Se o processo
cair (ou Ctrl+C), rode o mesmo comando: os ids já salvos são pulados.
Questões que falharam não são gravadas e são tentadas de novo.
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from prompt_complexity import analisar_complexidade_prompt

# ---------------------------------------------------------------------------
# Configurações avaliadas. Fixas durante todo o experimento: se mudar algo
# aqui, mude também a VERSAO_EXPERIMENTO e registre na documentação.
# ---------------------------------------------------------------------------
VERSAO_EXPERIMENTO = "piloto-v3"

# Temperatura e nível de raciocínio ficam no PADRÃO de cada modelo: para a
# família Gemini 3, o Google recomenda manter a temperatura padrão (1.0).
# Os tokens de raciocínio usados são registrados em cada chamada.
CONFIGURACOES = {
    "A": {"modelo": "gemini-3.5-flash-lite"},
    "B": {"modelo": "gemini-3.6-flash"},
}

# Preço do plano pago (US$ por 1 milhão de tokens), usado para calcular o
# custo EQUIVALENTE mesmo rodando no plano gratuito. Fonte:
# https://ai.google.dev/gemini-api/docs/pricing (consultado em 22/11/2026).
# Atenção: o preço do 3.6 Flash dobra em 01/01/2027; usamos o vigente na coleta.
# (O 3.8 Flash foi descartado no piloto: retornava 503 por excesso de demanda.)
# Tokens de raciocínio ("thinking") são cobrados como saída.
PRECOS_USD_POR_MILHAO = {
    "gemini-3.5-flash-lite": {"entrada": 0.30, "saida": 2.50},
    "gemini-3.6-flash": {"entrada": 0.75, "saida": 3.75},
}

# Instrução padrão enviada antes de cada questão. O modelo pode raciocinar,
# mas precisa terminar com uma linha "Resposta: X" para a correção automática.
INSTRUCAO = (
    "Resolva a questão de múltipla escolha abaixo. "
    "Ao final, escreva em uma linha separada exatamente no formato "
    "'Resposta: X', em que X é a letra da alternativa correta.\n\n"
)

COLUNAS_ENTRADA = ["id", "area", "fonte", "enunciado", "gabarito"]
METRICAS_CONFIG = [
    "modelo",
    "status",
    "resposta_letra",
    "acertou",
    "tokens_entrada",
    "tokens_saida",
    "tokens_raciocinio",
    "custo_usd",
    "latencia_s",
    "energia_kwh",
    "agua_l",
    "carbono_kg",
    "resposta_texto",
]
COLUNAS_SAIDA = (
    [
        "id", "area", "fonte", "executor", "coletado_em_utc",
        "versao_experimento", "gabarito",
        # características disponíveis ANTES da execução (entrada do ML)
        "enunciado", "qtd_palavras", "qtd_caracteres",
        "complexidade_pontuacao", "complexidade_nivel",
    ]
    + [f"{m}_A" for m in METRICAS_CONFIG]
    + [f"{m}_B" for m in METRICAS_CONFIG]
    + ["rotulo", "economia_usd", "pct_economia_custo"]
)

# Aceita "Resposta: C", "**Resposta:** C", "Resposta: **C**", "Resposta: (C)".
# Exige os dois-pontos e uma letra isolada, para não confundir com palavras
# como "a resposta correta é..." no meio do raciocínio.
PADRAO_RESPOSTA = re.compile(
    r"resposta\W{0,4}:\W{0,4}([A-E])(?![A-Za-zÀ-ÿ])",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def extrair_letra(texto):
    """Devolve a última letra A-E após 'Resposta:', ou None se não achar."""
    if not texto:
        return None
    achados = PADRAO_RESPOSTA.findall(texto)
    return achados[-1].upper() if achados else None


def calcular_custo(modelo, tokens_entrada, tokens_saida_total):
    precos = PRECOS_USD_POR_MILHAO.get(modelo)
    if precos is None:
        return None
    return (
        tokens_entrada * precos["entrada"]
        + tokens_saida_total * precos["saida"]
    ) / 1_000_000


def definir_rotulo(a, b):
    """Aplica a tabela de rótulos do README.

    economica_suficiente  -> A acertou e custou menos que B
    referencia_necessaria -> A errou e B acertou
    nao_atendido          -> nenhuma acertou
    inconclusivo          -> faltou resposta legível ou custo (revisar)
    """
    if a["acertou"] is None:
        return "inconclusivo"
    if a["acertou"]:
        # Se A acertou, o resultado de B só importa para comparar o custo.
        if a["custo_usd"] is None or b["custo_usd"] is None:
            return "inconclusivo"
        return "economica_suficiente" if a["custo_usd"] < b["custo_usd"] else "inconclusivo"
    if b["acertou"] is None:
        return "inconclusivo"
    return "referencia_necessaria" if b["acertou"] else "nao_atendido"


def valor_impacto(impacto):
    """Extrai número de um impacto do EcoLogits (usa o ponto médio de faixas)."""
    if impacto is None:
        return None
    v = getattr(impacto, "value", None)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    lo, hi = getattr(v, "min", None), getattr(v, "max", None)
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2
    return None


def extrair_impactos(resposta):
    imp = getattr(resposta, "impacts", None)
    if imp is None:
        return None, None, None
    return (
        valor_impacto(getattr(imp, "energy", None)),
        valor_impacto(getattr(imp, "wcf", None)),
        valor_impacto(getattr(imp, "gwp", None)),
    )


def chamar_config(client, config, questao, gabarito, tentativas=5):
    """Envia a questão a uma configuração (com retry) e corrige a resposta."""

    modelo = config["modelo"]
    prompt = INSTRUCAO + questao
    ultimo_erro = None
    for i in range(tentativas):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model=modelo,
                contents=prompt,
            )
            latencia = time.perf_counter() - t0
            uso = resp.usage_metadata
            entrada = uso.prompt_token_count or 0
            saida = uso.candidates_token_count or 0
            raciocinio = getattr(uso, "thoughts_token_count", None) or 0
            texto = (getattr(resp, "text", None) or "").strip()
            letra = extrair_letra(texto)
            energia, agua, carbono = extrair_impactos(resp)
            return {
                "modelo": modelo,
                "status": "sucesso" if letra else "sem_letra",
                "resposta_letra": letra,
                "acertou": (letra == gabarito) if letra else None,
                "tokens_entrada": entrada,
                "tokens_saida": saida,
                "tokens_raciocinio": raciocinio,
                "custo_usd": calcular_custo(modelo, entrada, saida + raciocinio),
                "latencia_s": round(latencia, 4),
                "energia_kwh": energia,
                "agua_l": agua,
                "carbono_kg": carbono,
                "resposta_texto": texto,
            }
        except Exception as e:  # noqa: BLE001 - qualquer falha de API
            ultimo_erro = e
            espera = min(2 ** (i + 1) * 5, 60)
            print(f"    ! {modelo}: {type(e).__name__}: {e}. "
                  f"Nova tentativa em {espera}s [{i + 1}/{tentativas}]",
                  file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError(f"{modelo} falhou após {tentativas} tentativas: {ultimo_erro}")


def montar_linha(questao, executor, a, b):
    complexidade = analisar_complexidade_prompt(questao["enunciado"])
    linha = {
        "id": questao["id"],
        "area": questao["area"],
        "fonte": questao["fonte"],
        "executor": executor,
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_experimento": VERSAO_EXPERIMENTO,
        "gabarito": questao["gabarito"],
        "enunciado": questao["enunciado"],
        "qtd_palavras": complexidade["metricas"]["quantidade_palavras"],
        "qtd_caracteres": complexidade["metricas"]["quantidade_caracteres"],
        "complexidade_pontuacao": complexidade["pontuacao"],
        "complexidade_nivel": complexidade["nivel"],
    }
    for m in METRICAS_CONFIG:
        linha[f"{m}_A"] = a[m]
        linha[f"{m}_B"] = b[m]

    linha["rotulo"] = definir_rotulo(a, b)
    if a["custo_usd"] is not None and b["custo_usd"] is not None:
        linha["economia_usd"] = b["custo_usd"] - a["custo_usd"]
        linha["pct_economia_custo"] = (
            round((b["custo_usd"] - a["custo_usd"]) / b["custo_usd"] * 100, 2)
            if b["custo_usd"] else None
        )
    else:
        linha["economia_usd"] = linha["pct_economia_custo"] = None
    return linha


def ids_ja_processados(saida):
    if not saida.exists() or saida.stat().st_size == 0:
        return set()
    try:
        return set(pd.read_csv(saida, usecols=["id"])["id"].astype(str))
    except Exception:  # noqa: BLE001
        return set()


def carregar_questoes(caminho):
    df = pd.read_csv(caminho, dtype=str).fillna("")
    faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    if faltando:
        sys.exit(f"Erro: colunas ausentes no CSV de entrada: {faltando}")
    df["gabarito"] = df["gabarito"].str.strip().str.upper()
    invalidos = df[~df["gabarito"].isin(list("ABCDE"))]
    if not invalidos.empty:
        sys.exit(f"Erro: gabarito inválido nos ids {list(invalidos['id'])} (use A a E).")
    if df["id"].duplicated().any():
        sys.exit("Erro: há ids repetidos no CSV de entrada.")
    return df


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Coletor do piloto TokenWise (A x B).")
    ap.add_argument("--entrada", required=True, help="CSV de questões deste executor")
    ap.add_argument("--saida", default=None,
                    help="CSV de saída (padrão: resultado_<executor>.csv)")
    ap.add_argument("--executor", required=True,
                    help="nome ou apelido de quem está rodando (não a chave!)")
    ap.add_argument("--pausa", type=float, default=None,
                    help="segundos entre chamadas (padrão: SLEEP_ENTRE_CHAMADAS ou 4.5)")
    ap.add_argument("--sem-ecologits", action="store_true",
                    help="não estimar energia/água/carbono")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "cole_sua_chave_aqui":
        sys.exit("Erro: defina GEMINI_API_KEY no arquivo .env.")
    pausa = args.pausa if args.pausa is not None else float(
        os.getenv("SLEEP_ENTRE_CHAMADAS", "4.5"))

    entrada = Path(args.entrada)
    saida = Path(args.saida or f"resultado_{args.executor}.csv")
    if not entrada.exists():
        sys.exit(f"Erro: arquivo de entrada não encontrado: {entrada}")
    questoes = carregar_questoes(entrada)

    if not args.sem_ecologits:
        from ecologits import EcoLogits
        EcoLogits.init(providers=["google_genai"])
    from google import genai
    client = genai.Client(api_key=api_key)

    feitos = ids_ja_processados(saida)
    pendentes = questoes[~questoes["id"].astype(str).isin(feitos)]
    print(f"Executor: {args.executor} | A: {CONFIGURACOES['A']['modelo']} | "
          f"B: {CONFIGURACOES['B']['modelo']}")
    print(f"Questões: {len(questoes)} | já feitas: {len(feitos)} | "
          f"pendentes: {len(pendentes)}")

    novo = not saida.exists() or saida.stat().st_size == 0
    contagem = {}
    falhas = 0
    with open(saida, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_SAIDA)
        if novo:
            writer.writeheader()
            f.flush()
        try:
            for n, (_, q) in enumerate(pendentes.iterrows(), 1):
                print(f"[{n}/{len(pendentes)}] id={q['id']} ({q['area']})")
                try:
                    a = chamar_config(client, CONFIGURACOES["A"], q["enunciado"], q["gabarito"])
                    time.sleep(pausa)
                    b = chamar_config(client, CONFIGURACOES["B"], q["enunciado"], q["gabarito"])
                    time.sleep(pausa)
                except RuntimeError as e:
                    print(f"    x id={q['id']} não gravado: {e}", file=sys.stderr)
                    falhas += 1
                    continue

                linha = montar_linha(q, args.executor, a, b)
                writer.writerow(linha)
                f.flush()
                os.fsync(f.fileno())
                contagem[linha["rotulo"]] = contagem.get(linha["rotulo"], 0) + 1
                print(f"    A={a['resposta_letra']} B={b['resposta_letra']} "
                      f"gabarito={q['gabarito']} -> {linha['rotulo']}")
        except KeyboardInterrupt:
            print("\nInterrompido. Rode o mesmo comando para continuar de onde parou.")

    print(f"\nConcluído. Falhas: {falhas}. Saída: {saida}")
    print("Rótulos nesta execução:")
    for rotulo, qtd in sorted(contagem.items()):
        print(f"  {rotulo}: {qtd}")


if __name__ == "__main__":
    main()