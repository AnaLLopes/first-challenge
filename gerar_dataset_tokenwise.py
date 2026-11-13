#!/usr/bin/env python3
"""TokenWise - gerador de dataset.

Lê pares de prompts (original x otimizado), envia ambos ao Gemini, mede
latência, tokens e impacto ambiental (EcoLogits) e grava tudo em um CSV,
uma linha por par, com salvamento incremental e retomada automática.

Uso:
    python gerar_dataset_tokenwise.py \
        --entrada prompts_entrada_exemplo.csv --saida dataset_tokenwise.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ecologits import EcoLogits
from google import genai
from google.genai import types

COLUNAS_ENTRADA = ["id", "tarefa", "prompt_original", "prompt_otimizado"]
METRICAS = [
    "latencia_s",
    "tokens_entrada",
    "tokens_saida",
    "energia_kwh",
    "agua_l",
    "carbono_kg",
]
COLUNAS_SAIDA = (
    ["id", "tarefa", "modelo", "prompt_original", "prompt_otimizado"]
    + [f"{m}_original" for m in METRICAS]
    + [f"{m}_otimizado" for m in METRICAS]
    + [
        "pct_reducao_tokens_entrada",
        "pct_reducao_tokens_saida",
        "pct_reducao_tokens_total",
        "economia_energia_kwh",
        "economia_agua_l",
        "economia_carbono_kg",
        "pct_economia_energia",
        "pct_economia_agua",
        "pct_economia_carbono",
    ]
)


def valor_impacto(impacto):
    """Extrai um número de um impacto do EcoLogits.

    O EcoLogits pode devolver um valor simples ou um intervalo (min/max);
    nesse caso usamos o ponto médio. Retorna None se não houver dado.
    """
    if impacto is None:
        return None
    v = getattr(impacto, "value", None)
    if v is not None:
        return float(v)
    lo, hi = getattr(impacto, "min", None), getattr(impacto, "max", None)
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2
    return None


def extrair_impactos(resposta):
    """Retorna (energia_kwh, agua_l, carbono_kg) da resposta, se disponível."""
    imp = getattr(resposta, "impacts", None)
    if imp is None or not hasattr(imp, "energy"):
        return None, None, None
    energia = valor_impacto(getattr(imp, "energy", None))
    carbono = valor_impacto(getattr(imp, "gwp", None))
    agua = valor_impacto(getattr(imp, "wcf", None))  # consumo de água (L)
    return energia, agua, carbono


def chamar_modelo(client, modelo, prompt, tentativas=5):
    """Chama o Gemini com retry e backoff exponencial. Retorna dict de métricas."""
    ultimo_erro = None
    for i in range(tentativas):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(model=modelo, contents=prompt)
            latencia = time.perf_counter() - t0

            uso = resp.usage_metadata
            # tokens de "thinking" (modelos 2.5) também são cobrados como saída
            saida = (uso.candidates_token_count or 0) + (
                getattr(uso, "thoughts_token_count", None) or 0
            )
            energia, agua, carbono = extrair_impactos(resp)
            return {
                "latencia_s": round(latencia, 4),
                "tokens_entrada": uso.prompt_token_count or 0,
                "tokens_saida": saida,
                "energia_kwh": energia,
                "agua_l": agua,
                "carbono_kg": carbono,
            }
        except Exception as e:  # noqa: BLE001 - queremos tratar qualquer falha de API
            ultimo_erro = e
            espera = min(2 ** (i + 1) * 5, 60)  # 10s, 20s, 40s, 60s...
            print(f"    ! erro ({type(e).__name__}): {e}. Retry em {espera}s "
                  f"[{i + 1}/{tentativas}]", file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError(f"Falha após {tentativas} tentativas: {ultimo_erro}")


def pct_reducao(orig, otim):
    if orig is None or otim is None or orig == 0:
        return None
    return round((orig - otim) / orig * 100, 2)


def diff(orig, otim):
    if orig is None or otim is None:
        return None
    return orig - otim


def montar_linha(par, modelo, o, p):
    linha = {
        "id": par["id"],
        "tarefa": par["tarefa"],
        "modelo": modelo,
        "prompt_original": par["prompt_original"],
        "prompt_otimizado": par["prompt_otimizado"],
    }
    for m in METRICAS:
        linha[f"{m}_original"] = o[m]
        linha[f"{m}_otimizado"] = p[m]
    tot_o = o["tokens_entrada"] + o["tokens_saida"]
    tot_p = p["tokens_entrada"] + p["tokens_saida"]
    linha.update({
        "pct_reducao_tokens_entrada": pct_reducao(o["tokens_entrada"], p["tokens_entrada"]),
        "pct_reducao_tokens_saida": pct_reducao(o["tokens_saida"], p["tokens_saida"]),
        "pct_reducao_tokens_total": pct_reducao(tot_o, tot_p),
        "economia_energia_kwh": diff(o["energia_kwh"], p["energia_kwh"]),
        "economia_agua_l": diff(o["agua_l"], p["agua_l"]),
        "economia_carbono_kg": diff(o["carbono_kg"], p["carbono_kg"]),
        "pct_economia_energia": pct_reducao(o["energia_kwh"], p["energia_kwh"]),
        "pct_economia_agua": pct_reducao(o["agua_l"], p["agua_l"]),
        "pct_economia_carbono": pct_reducao(o["carbono_kg"], p["carbono_kg"]),
    })
    return linha


def ids_ja_processados(saida: Path):
    if not saida.exists() or saida.stat().st_size == 0:
        return set()
    try:
        return set(pd.read_csv(saida, usecols=["id"])["id"].astype(str))
    except Exception:  # noqa: BLE001
        return set()


def main():
    ap = argparse.ArgumentParser(description="Gera o dataset do TokenWise.")
    ap.add_argument("--entrada", default="prompts_entrada_exemplo.csv")
    ap.add_argument("--saida", default="dataset_tokenwise.csv")
    ap.add_argument("--modelo", default=None, help="padrão: GEMINI_MODEL ou gemini-2.5-flash")
    ap.add_argument("--pausa", type=float, default=None,
                    help="segundos entre chamadas (padrão: SLEEP_ENTRE_CHAMADAS ou 4.5)")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Erro: defina GEMINI_API_KEY no arquivo .env (veja .env.example).")
    modelo = args.modelo or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    pausa = args.pausa if args.pausa is not None else float(os.getenv("SLEEP_ENTRE_CHAMADAS", "4.5"))

    entrada, saida = Path(args.entrada), Path(args.saida)
    if not entrada.exists():
        sys.exit(f"Erro: arquivo de entrada não encontrado: {entrada}")
    df = pd.read_csv(entrada, dtype=str).fillna("")
    faltando = [c for c in COLUNAS_ENTRADA if c not in df.columns]
    if faltando:
        sys.exit(f"Erro: colunas ausentes no CSV de entrada: {faltando}")

    # Instrumenta o SDK do Gemini para que cada resposta traga `.impacts`
    EcoLogits.init(providers=["google_genai"])
    client = genai.Client(api_key=api_key)

    feitos = ids_ja_processados(saida)
    pendentes = df[~df["id"].astype(str).isin(feitos)]
    print(f"Modelo: {modelo} | total: {len(df)} | já feitos: {len(feitos)} | "
          f"pendentes: {len(pendentes)}")

    novo_arquivo = not saida.exists() or saida.stat().st_size == 0
    ok = falhas = 0
    with open(saida, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_SAIDA)
        if novo_arquivo:
            writer.writeheader()
            f.flush()

        try:
            for n, (_, par) in enumerate(pendentes.iterrows(), 1):
                print(f"[{n}/{len(pendentes)}] id={par['id']} ({par['tarefa']})")
                try:
                    o = chamar_modelo(client, modelo, par["prompt_original"])
                    time.sleep(pausa)
                    p = chamar_modelo(client, modelo, par["prompt_otimizado"])
                    time.sleep(pausa)
                except RuntimeError as e:
                    # não grava a linha: será tentada de novo na próxima execução
                    print(f"    ✗ id={par['id']} ignorado: {e}", file=sys.stderr)
                    falhas += 1
                    continue

                writer.writerow(montar_linha(par, modelo, o, p))
                f.flush()
                os.fsync(f.fileno())
                ok += 1
        except KeyboardInterrupt:
            print("\nInterrompido. Rode de novo para retomar de onde parou.")

    print(f"Concluído: {ok} salvos, {falhas} falharam. Saída: {saida}")


if __name__ == "__main__":
    main()
