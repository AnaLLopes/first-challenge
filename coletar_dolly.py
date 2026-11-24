#!/usr/bin/env python3
"""TokenWise - coletor do piloto para prompts ABERTOS (Dolly 15k em PT-BR).

Adaptação do coletar_piloto.py (que NÃO é alterado): reaproveita o mesmo
schema de saída (COLUNAS_SAIDA), as mesmas configurações A/B, a mesma
tabela de preços, o mesmo cálculo de rótulo e a mesma extração do EcoLogits.

Diferenças em relação ao coletar_piloto.py:
  - entrada: meus_prompts_traduzidos.csv (id, tarefa, prompt_original)
  - o prompt é enviado direto aos modelos, sem template de múltipla escolha
  - não há gabarito: um juiz (gemini-3.6-flash, resposta em JSON) com ordem
    anonimizada/sorteada (RESPOSTA_1 vs RESPOSTA_2) avalia as respostas e
    preenche acertou_A, acertou_B e rotulo

    A = configuração econômica     (gemini-3.5-flash-lite)
    B = configuração de referência (gemini-3.6-flash)

Uso:
    python coletar_dolly.py --limite 5
    python coletar_dolly.py                # padrão: 200 linhas gravadas
    python coletar_dolly.py --sem-juiz     # só coleta A e B (julgamento depois)
    python coletar_dolly.py --sem-juiz --inicio 1 --fim 100 --executor gabriel --saida piloto_dolly_gabriel.csv

Modo --sem-juiz: não chama o juiz (economiza a cota diária do gemini-3.6-flash).
As linhas saem com acertou_A/acertou_B vazios e rotulo "inconclusivo" (pendente);
depois, o julgamento em lote preenche acertou_A, acertou_B e rotulo.

Cota diária (429 "PerDay"): o script para na hora, sem gravar a linha em curso,
e avisa. Não adianta insistir: a cota só volta no dia seguinte. Basta rodar o
mesmo comando depois, que ele retoma de onde parou.

Retomada: cada prompt é gravado assim que A, B e o juiz terminam. Se o
processo cair (ou Ctrl+C), rode o mesmo comando: os ids já salvos são
pulados. Prompts que falharam não são gravados e são tentados de novo.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from prompt_complexity import analisar_complexidade_prompt

# Reaproveita do coletar_piloto.py tudo o que define o schema e as regras de
# negócio, para garantir que os dois CSVs fiquem compatíveis.
from coletar_piloto import (
    COLUNAS_SAIDA,
    CONFIGURACOES,
    METRICAS_CONFIG,
    calcular_custo,
    definir_rotulo,
    extrair_impactos,
)

PASTA = Path(__file__).resolve().parent
VERSAO_EXPERIMENTO = "piloto-dolly-v1"
MODELO_JUIZ = "gemini-3.6-flash"
PREFIXO_ID = "dolly_"
FONTE = "dolly-15k"
GABARITO_ABERTA = "ABERTA"

COLUNAS_ENTRADA = ["id", "tarefa", "prompt_original"]


class CotaDiariaEsgotada(Exception):
    """A cota diária (free tier) acabou: tentar de novo hoje é inútil."""


def _cota_diaria_esgotada(erro):
    """True para 429 de limite POR DIA (ex.: GenerateRequestsPerDayPerProject...).

    429 por minuto (rate limit) continua passando pelo retry normal.
    """
    texto = str(erro).lower()
    eh_429 = "429" in texto or "resource_exhausted" in texto
    return eh_429 and ("perday" in texto or "per day" in texto)

INSTRUCAO_JUIZ = (
    "Você é um avaliador imparcial de respostas de modelos de linguagem. "
    "Abaixo estão um PROMPT do usuário e duas respostas anônimas "
    "(RESPOSTA_1 e RESPOSTA_2), apresentadas em ordem aleatória. Avalie:\n"
    "1. r1_atende: a RESPOSTA_1 atende bem ao prompt (correta, completa, "
    "relevante e no idioma/formato pedido)?\n"
    "2. r2_atende: a RESPOSTA_2 atende bem ao prompt?\n"
    "3. melhor: indique '1' se a RESPOSTA_1 for claramente superior em "
    "correção ou completude essencial; '2' se a RESPOSTA_2 for claramente "
    "superior; ou 'empate' se ambas tiverem qualidade equivalente.\n"
    "Regras importantes:\n"
    "- NÃO prefira uma resposta apenas por ser mais longa ou prolixa se "
    "uma resposta concisa já atender completa e corretamente ao enunciado.\n"
    "- Seja criterioso: respostas erradas, cortadas, vazias ou que ignoram "
    "o pedido NÃO atendem.\n"
    "Responda ESTRITAMENTE com um objeto JSON com as chaves booleanas "
    "'r1_atende' e 'r2_atende', a chave 'melhor' ('1', '2' ou 'empate') "
    "e a chave de texto curto 'justificativa'."
)


# ---------------------------------------------------------------------------
# Chamadas aos modelos
# ---------------------------------------------------------------------------
def chamar_config(client, config, enunciado, tentativas=8):
    """Envia o prompt a uma configuração (com retry/backoff + jitter)."""
    modelo = config["modelo"]
    ultimo_erro = None
    for i in range(tentativas):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(model=modelo, contents=enunciado)
            latencia = time.perf_counter() - t0
            uso = resp.usage_metadata
            entrada = uso.prompt_token_count or 0
            saida = uso.candidates_token_count or 0
            raciocinio = getattr(uso, "thoughts_token_count", None) or 0
            texto = (getattr(resp, "text", None) or "").strip()
            energia, agua, carbono = extrair_impactos(resp)
            return {
                "modelo": modelo,
                "status": "sucesso" if texto else "sem_resposta",
                "resposta_letra": "",
                "acertou": None,  # preenchido pelo juiz
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
        except Exception as e:  # noqa: BLE001 - qualquer falha de API (429, 503...)
            if _cota_diaria_esgotada(e):
                raise CotaDiariaEsgotada(f"{modelo}: {e}") from e
            ultimo_erro = e
            espera = min(2 ** (i + 1) * 5, 60) + random.uniform(0, 5)
            print(f"    ! {modelo}: {type(e).__name__}: {e}. "
                  f"Nova tentativa em {espera:.1f}s [{i + 1}/{tentativas}]",
                  file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError(f"{modelo} falhou após {tentativas} tentativas: {ultimo_erro}")


def _limpar_json(texto):
    texto = (texto or "").strip()
    if texto.startswith("```"):
        texto = texto.strip("`").strip()
        if texto.lower().startswith("json"):
            texto = texto[4:].strip()
    return texto


def _como_bool(valor):
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, str):
        v = valor.strip().lower()
        if v in ("true", "sim", "yes"):
            return True
        if v in ("false", "nao", "não", "no"):
            return False
    return None


def julgar(client, enunciado, a, b, tentativas=8):
    """Juiz (gemini-3.6-flash, JSON) com ordem sorteada (RESPOSTA_1/2).

    Retorna (acertou_A, acertou_B), ou (None, None) se o juiz falhar: o rótulo
    fica 'inconclusivo' e a linha pode ser revisada depois. Resposta vazia
    nunca é aprovada.
    """
    from google.genai import types

    if not a["resposta_texto"] and not b["resposta_texto"]:
        return None, None

    ordem_invertida = random.choice([False, True])
    r1_texto = b["resposta_texto"] if ordem_invertida else a["resposta_texto"]
    r2_texto = a["resposta_texto"] if ordem_invertida else b["resposta_texto"]

    conteudo = (
        f"{INSTRUCAO_JUIZ}\n\n"
        f"=== PROMPT ===\n{enunciado}\n\n"
        f"=== RESPOSTA_1 ===\n{r1_texto or '(vazia)'}\n\n"
        f"=== RESPOSTA_2 ===\n{r2_texto or '(vazia)'}\n"
    )
    for i in range(tentativas):
        try:
            resp = client.models.generate_content(
                model=MODELO_JUIZ,
                contents=conteudo,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            dados = json.loads(_limpar_json(getattr(resp, "text", None)))
            r1_atende = _como_bool(dados.get("r1_atende"))
            r2_atende = _como_bool(dados.get("r2_atende"))
            melhor = str(dados.get("melhor", "")).strip().lower()
            if r1_atende is None or r2_atende is None or melhor not in ("1", "2", "empate"):
                raise ValueError(f"JSON do juiz incompleto/inválido: {dados}")

            if ordem_invertida:
                a_atende, b_atende = r2_atende, r1_atende
                b_claramente_melhor = (melhor == "1")
            else:
                a_atende, b_atende = r1_atende, r2_atende
                b_claramente_melhor = (melhor == "2")

            if not a["resposta_texto"]:
                a_atende = False
            if not b["resposta_texto"]:
                b_atende = False

            # A só conta como suficiente se atende E B não foi claramente superior.
            acertou_a = a_atende and (not b_claramente_melhor)
            return acertou_a, b_atende
        except Exception as e:  # noqa: BLE001
            if _cota_diaria_esgotada(e):
                raise CotaDiariaEsgotada(f"juiz ({MODELO_JUIZ}): {e}") from e
            espera = min(4 * (2 ** i), 60) + random.uniform(0, 5)
            print(f"    ! juiz: {type(e).__name__}: {e}. "
                  f"Nova tentativa em {espera:.1f}s [{i + 1}/{tentativas}]",
                  file=sys.stderr)
            time.sleep(espera)
    print("    ! juiz falhou: linha ficará 'inconclusivo'.", file=sys.stderr)
    return None, None


# ---------------------------------------------------------------------------
# Montagem da linha (mesmo schema do coletar_piloto.py)
# ---------------------------------------------------------------------------
def montar_linha(q, executor, a, b):
    complexidade = analisar_complexidade_prompt(q["prompt_original"])
    linha = {
        "id": f"{PREFIXO_ID}{q['id']}",
        "area": q["tarefa"],
        "fonte": FONTE,
        "executor": executor,
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_experimento": VERSAO_EXPERIMENTO,
        "gabarito": GABARITO_ABERTA,
        "enunciado": q["prompt_original"],
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


def carregar_prompts(caminho):
    df = pd.read_csv(caminho, dtype=str).fillna("")
    faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    if faltando:
        sys.exit(f"Erro: colunas ausentes no CSV de entrada: {faltando}")
    df = df[df["prompt_original"].str.strip() != ""]
    if df["id"].duplicated().any():
        sys.exit("Erro: há ids repetidos no CSV de entrada.")
    return df


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Coletor do piloto TokenWise (Dolly, A x B).")
    ap.add_argument("--entrada", default=str(PASTA / "meus_prompts_traduzidos.csv"))
    ap.add_argument("--saida", default=str(PASTA / "piloto_dolly_coletado.csv"))
    ap.add_argument("--limite", type=int, default=200,
                    help="quantas linhas novas gravar nesta execução (padrão: 200; <=0 para sem limite)")
    ap.add_argument("--inicio", type=int, default=None,
                    help="id inteiro inicial da faixa no CSV de entrada (inclusivo)")
    ap.add_argument("--fim", type=int, default=None,
                    help="id inteiro final da faixa no CSV de entrada (inclusivo)")
    ap.add_argument("--executor", default="dolly",
                    help="nome/apelido de quem está rodando (não a chave!)")
    ap.add_argument("--pausa", type=float, default=None,
                    help="segundos entre chamadas (padrão: SLEEP_ENTRE_CHAMADAS ou 4.5)")
    ap.add_argument("--sem-ecologits", action="store_true",
                    help="não estimar energia/água/carbono")
    ap.add_argument("--sem-juiz", action="store_true",
                    help="não chama o juiz: grava só A e B (julgue depois em lote)")
    args = ap.parse_args()

    load_dotenv(PASTA / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "cole_sua_chave_aqui":
        sys.exit("Erro: defina GEMINI_API_KEY no arquivo .env.")
    pausa = args.pausa if args.pausa is not None else float(
        os.getenv("SLEEP_ENTRE_CHAMADAS", "4.5"))

    if args.inicio is not None and args.fim is not None and args.inicio > args.fim:
        sys.exit(f"Erro: --inicio ({args.inicio}) não pode ser maior que --fim ({args.fim}).")

    entrada, saida = Path(args.entrada), Path(args.saida)
    if not entrada.exists():
        sys.exit(f"Erro: arquivo de entrada não encontrado: {entrada}")
    prompts = carregar_prompts(entrada)

    if args.inicio is not None or args.fim is not None:
        ids_num = pd.to_numeric(prompts["id"], errors="coerce")
        mascara = ids_num.notna()
        if args.inicio is not None:
            mascara &= ids_num >= args.inicio
        if args.fim is not None:
            mascara &= ids_num <= args.fim
        prompts = prompts[mascara]

    if not args.sem_ecologits:
        from ecologits import EcoLogits
        EcoLogits.init(providers=["google_genai"])
    from google import genai
    client = genai.Client(api_key=api_key)

    feitos = ids_ja_processados(saida)
    ids_prefixados = prompts["id"].map(lambda i: f"{PREFIXO_ID}{i}")
    pendentes = prompts[~ids_prefixados.isin(feitos)]
    meta = args.limite if (args.limite is not None and args.limite > 0) else len(pendentes)
    meta = min(meta, len(pendentes))

    print(f"Executor: {args.executor} | A: {CONFIGURACOES['A']['modelo']} | "
          f"B: {CONFIGURACOES['B']['modelo']} | "
          f"Juiz: {'DESLIGADO (--sem-juiz)' if args.sem_juiz else MODELO_JUIZ}")
    print(f"Prompts: {len(prompts)} | já feitos: {len(feitos)} | "
          f"pendentes: {len(pendentes)} | meta de gravação: {meta}")

    novo = not saida.exists() or saida.stat().st_size == 0
    contagem = {}
    gravados = 0
    falhas = 0
    with open(saida, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_SAIDA)
        if novo:
            writer.writeheader()
            f.flush()
        try:
            for _, q in pendentes.iterrows():
                if gravados >= meta:
                    break
                print(f"[gravados={gravados}/{meta}] {PREFIXO_ID}{q['id']} ({q['tarefa']})")
                try:
                    a = chamar_config(client, CONFIGURACOES["A"], q["prompt_original"])
                    time.sleep(pausa)
                    b = chamar_config(client, CONFIGURACOES["B"], q["prompt_original"])
                    time.sleep(pausa)
                except RuntimeError as e:
                    print(f"    x {PREFIXO_ID}{q['id']} não gravado: {e}", file=sys.stderr)
                    falhas += 1
                    continue

                if args.sem_juiz:
                    a["acertou"], b["acertou"] = None, None  # julgamento em lote depois
                else:
                    a["acertou"], b["acertou"] = julgar(client, q["prompt_original"], a, b)
                    time.sleep(pausa)

                linha = montar_linha(q, args.executor, a, b)
                writer.writerow(linha)
                f.flush()
                os.fsync(f.fileno())
                gravados += 1
                contagem[linha["rotulo"]] = contagem.get(linha["rotulo"], 0) + 1
                print(f"    acertou_A={a['acertou']} acertou_B={b['acertou']} "
                      f"-> {linha['rotulo']} ({gravados}/{meta})")
        except KeyboardInterrupt:
            print("\nInterrompido. Rode o mesmo comando para continuar de onde parou.")
        except CotaDiariaEsgotada as e:
            print("\n!!! COTA DIÁRIA ESGOTADA - coleta encerrada de forma segura.",
                  file=sys.stderr)
            print(f"    {str(e)[:300]}", file=sys.stderr)
            print("    A linha em curso NÃO foi gravada. A cota só volta no dia "
                  "seguinte: rode o mesmo comando depois (ou use outra chave).",
                  file=sys.stderr)

    print(f"\nConcluído. Gravados: {gravados} | Falhas: {falhas}. Saída: {saida}")
    print("Rótulos nesta execução:")
    for rotulo, qtd in sorted(contagem.items()):
        print(f"  {rotulo}: {qtd}")


if __name__ == "__main__":
    main()
