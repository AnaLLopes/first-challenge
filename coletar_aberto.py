#!/usr/bin/env python3
"""TokenWise - coletor A x B para prompts ABERTOS (sem gabarito), com juiz.

Feito para os prompts do WildChat (arquivo gerado por `wildchat.py amostra`), mas serve
para qualquer CSV com as colunas `id` e `prompt_original`.

    A = configuração econômica     (gemini-3.5-flash-lite)   -> definidas em coletar_piloto.py
    B = configuração de referência (gemini-3.6-flash)
    Juiz = modelo da Mistral (padrão: mistral-large-latest), de outra empresa,
           para não "corrigir a própria prova"

Como funciona, para cada prompt:
  1. envia o prompt ao modelo A e ao modelo B (Gemini);
  2. envia ao juiz o prompt e as duas respostas, sem dizer qual é de qual modelo
     (a ordem é sorteada: RESPOSTA_1 / RESPOSTA_2);
  3. o juiz diz se cada resposta atende ao pedido e se uma é claramente melhor,
     com uma justificativa curta;
  4. grava tudo numa linha, com o rótulo.

Regra do rótulo (documentar no README):
  - A "atende" só se o juiz aprovar a resposta de A E a resposta de B não for claramente melhor;
  - economica_suficiente  -> A atende (e custa menos que B)
  - referencia_necessaria -> A não atende e B atende
  - nao_atendido          -> nenhuma atende
  - inconclusivo          -> juiz pendente ou com erro

Uso:
    python coletar_aberto.py --entrada wildchat_amostra_50.csv --executor charles
    python coletar_aberto.py --entrada wildchat_amostra_50.csv --executor charles --limite 5
    python coletar_aberto.py --entrada ... --executor ... --sem-juiz     # só A e B (julga depois)
    python coletar_aberto.py --saida resultado_aberto_charles.csv --julgar-pendentes

Chaves no .env:
    GEMINI_API_KEY=...
    MISTRAL_API_KEY=...

Retomada: cada prompt é gravado assim que termina. Se o processo cair (ou Ctrl+C), rode o
mesmo comando: os ids já gravados são pulados. Prompts que falharam em A ou B não são
gravados e são tentados de novo. Se a cota DIÁRIA do Gemini acabar, o script para e avisa.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from prompt_complexity import analisar_complexidade_prompt
from coletar_piloto import (
    COLUNAS_SAIDA,
    CONFIGURACOES,
    METRICAS_CONFIG,
    calcular_custo,
    definir_rotulo,
    extrair_impactos,
)

PASTA = Path(__file__).resolve().parent
VERSAO_EXPERIMENTO = "aberto-v1"
MODELO_JUIZ_PADRAO = "mistral-large-latest"
URL_MISTRAL = "https://api.mistral.ai/v1/chat/completions"
REGRA_ROTULO = "A atende = juiz aprova A e B não é claramente melhor"

# Colunas extras, depois das do coletar_piloto.py (os dois CSVs continuam compatíveis)
COLUNAS_EXTRAS = [
    "conversation_hash", "grupo", "tipo_correcao", "regra_rotulo",
    "finish_reason_A", "finish_reason_B",
    "juiz_modelo", "juiz_status", "juiz_ordem_invertida",
    "juiz_r1_atende", "juiz_r2_atende", "juiz_melhor",
    "juiz_a_atende", "juiz_b_atende", "juiz_b_claramente_melhor",
    "juiz_justificativa", "juiz_tokens_entrada", "juiz_tokens_saida",
]
COLUNAS = COLUNAS_SAIDA + COLUNAS_EXTRAS

INSTRUCAO_JUIZ = """Você é um avaliador imparcial de respostas de assistentes de IA.
Abaixo estão o PEDIDO de um usuário e duas respostas anônimas (RESPOSTA_1 e RESPOSTA_2),
em ordem aleatória. Avalie cada resposta de forma independente.

Uma resposta ATENDE ao pedido quando:
- faz o que foi pedido, de forma correta e completa no essencial;
- está no idioma e no formato pedidos (se o pedido não definir, no idioma do pedido);
- não foi cortada no meio nem ficou vazia.

Regras importantes:
- PREMISSA FALSA: se o pedido parte de um fato falso ou de algo que não existe, uma resposta
  que aceita a premissa e inventa detalhes NÃO atende; uma resposta que aponta o problema atende.
- FATOS INVENTADOS: datas, nomes, números ou eventos inventados fazem a resposta NÃO atender.
- RECUSA: se o pedido é legítimo e a resposta se recusa a fazê-lo, NÃO atende. Se o pedido não
  pode ser atendido como está (ex.: reproduzir uma letra de música protegida), uma recusa educada
  que oferece uma alternativa útil atende.
- TAMANHO: não prefira uma resposta só por ser mais longa. Uma resposta curta que atende
  completamente é tão boa quanto uma longa.
- CÓDIGO e CÁLCULO: verifique se a lógica e o resultado estão corretos.

Depois, indique se alguma resposta é CLARAMENTE superior em correção ou completude essencial:
"1", "2" ou "empate" (use "empate" quando a diferença for só de estilo ou tamanho).

Responda SOMENTE com um objeto JSON, exatamente com estas chaves:
{"r1_atende": true ou false, "r2_atende": true ou false, "melhor": "1" ou "2" ou "empate",
 "justificativa": "até 3 frases explicando a avaliação"}"""


class CotaDiariaEsgotada(Exception):
    """A cota diária do Gemini (plano gratuito) acabou: só volta no dia seguinte."""


def cota_diaria_esgotada(erro):
    texto = str(erro).lower()
    eh_429 = "429" in texto or "resource_exhausted" in texto
    return eh_429 and ("perday" in texto or "per day" in texto)


def espera_backoff(tentativa):
    return min(2 ** (tentativa + 1) * 5, 60) + random.uniform(0, 5)


# ---------------------------------------------------------------------------
# Modelos A e B (Gemini)
# ---------------------------------------------------------------------------
def chamar_config(client, config, prompt, tentativas=8):
    """Envia o prompt a uma configuração do Gemini, com novas tentativas em caso de erro."""
    modelo = config["modelo"]
    ultimo_erro = None
    for i in range(tentativas):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(model=modelo, contents=prompt)
            latencia = time.perf_counter() - t0
            uso = resp.usage_metadata
            entrada = uso.prompt_token_count or 0
            saida = uso.candidates_token_count or 0
            raciocinio = getattr(uso, "thoughts_token_count", None) or 0
            texto = (getattr(resp, "text", None) or "").strip()
            candidatos = getattr(resp, "candidates", None) or []
            finish = getattr(candidatos[0], "finish_reason", None) if candidatos else None
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
                "finish_reason": str(finish) if finish is not None else "",
            }
        except Exception as e:  # noqa: BLE001 - qualquer falha de API (429, 503...)
            if cota_diaria_esgotada(e):
                raise CotaDiariaEsgotada(f"{modelo}: {e}") from e
            ultimo_erro = e
            espera = espera_backoff(i)
            print(f"    ! {modelo}: {type(e).__name__}: {str(e)[:200]}. "
                  f"Nova tentativa em {espera:.0f}s [{i + 1}/{tentativas}]", file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError(f"{modelo} falhou após {tentativas} tentativas: {ultimo_erro}")


# ---------------------------------------------------------------------------
# Juiz (Mistral, pela API HTTP, sem biblioteca extra)
# ---------------------------------------------------------------------------
def chamar_mistral(chave, modelo, conteudo, timeout=180):
    corpo = json.dumps({
        "model": modelo,
        "messages": [{"role": "user", "content": conteudo}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode("utf-8")
    pedido = urllib.request.Request(
        URL_MISTRAL, data=corpo, method="POST",
        headers={"Authorization": f"Bearer {chave}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(pedido, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detalhe = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} da Mistral: {detalhe}") from e


def como_bool(valor):
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, str):
        v = valor.strip().lower()
        if v in ("true", "sim", "yes"):
            return True
        if v in ("false", "nao", "não", "no"):
            return False
    return None


def julgar(chave, modelo_juiz, prompt, resposta_a, resposta_b, tentativas=6):
    """Devolve um dicionário com o veredito do juiz (status 'ok' ou 'erro')."""
    resultado = {"juiz_modelo": modelo_juiz, "juiz_status": "erro"}
    if not resposta_a and not resposta_b:
        resultado.update(juiz_status="sem_respostas", juiz_a_atende=False, juiz_b_atende=False,
                         juiz_justificativa="As duas respostas estão vazias.")
        return resultado

    invertida = random.choice([False, True])
    r1, r2 = (resposta_b, resposta_a) if invertida else (resposta_a, resposta_b)
    conteudo = (f"{INSTRUCAO_JUIZ}\n\n=== PEDIDO ===\n{prompt}\n\n"
                f"=== RESPOSTA_1 ===\n{r1 or '(vazia)'}\n\n=== RESPOSTA_2 ===\n{r2 or '(vazia)'}\n")

    for i in range(tentativas):
        try:
            dados_api = chamar_mistral(chave, modelo_juiz, conteudo)
            texto = dados_api["choices"][0]["message"]["content"]
            veredito = json.loads(texto)
            r1_ok, r2_ok = como_bool(veredito.get("r1_atende")), como_bool(veredito.get("r2_atende"))
            melhor = str(veredito.get("melhor", "")).strip().lower()
            if r1_ok is None or r2_ok is None or melhor not in ("1", "2", "empate"):
                raise ValueError(f"resposta do juiz incompleta: {texto[:200]}")

            a_ok, b_ok = (r2_ok, r1_ok) if invertida else (r1_ok, r2_ok)
            b_melhor = (melhor == "1") if invertida else (melhor == "2")
            if not resposta_a:
                a_ok = False
            if not resposta_b:
                b_ok = False
            uso = dados_api.get("usage", {}) or {}
            resultado.update(
                juiz_status="ok", juiz_ordem_invertida=invertida,
                juiz_r1_atende=r1_ok, juiz_r2_atende=r2_ok, juiz_melhor=melhor,
                juiz_a_atende=a_ok, juiz_b_atende=b_ok, juiz_b_claramente_melhor=b_melhor,
                juiz_justificativa=str(veredito.get("justificativa", "")).strip(),
                juiz_tokens_entrada=uso.get("prompt_tokens"),
                juiz_tokens_saida=uso.get("completion_tokens"),
            )
            return resultado
        except Exception as e:  # noqa: BLE001
            espera = espera_backoff(i)
            print(f"    ! juiz ({modelo_juiz}): {type(e).__name__}: {str(e)[:200]}. "
                  f"Nova tentativa em {espera:.0f}s [{i + 1}/{tentativas}]", file=sys.stderr)
            time.sleep(espera)
    resultado["juiz_justificativa"] = "O juiz falhou em todas as tentativas."
    print("    ! juiz falhou: a linha fica 'inconclusivo' (use --julgar-pendentes depois).",
          file=sys.stderr)
    return resultado


def aplicar_veredito(linha, veredito):
    """Preenche acertou_A, acertou_B e o rótulo a partir do veredito do juiz."""
    linha.update(veredito)
    if veredito.get("juiz_status") in ("ok", "sem_respostas"):
        a_atende = bool(veredito.get("juiz_a_atende"))
        b_melhor = bool(veredito.get("juiz_b_claramente_melhor"))
        linha["acertou_A"] = a_atende and not b_melhor
        linha["acertou_B"] = bool(veredito.get("juiz_b_atende"))
    else:
        linha["acertou_A"] = linha["acertou_B"] = None
    linha["rotulo"] = definir_rotulo(
        {"acertou": linha["acertou_A"], "custo_usd": num(linha.get("custo_usd_A"))},
        {"acertou": linha["acertou_B"], "custo_usd": num(linha.get("custo_usd_B"))})
    return linha


def num(valor):
    try:
        v = float(valor)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Montagem da linha
# ---------------------------------------------------------------------------
def montar_linha(q, args, a, b):
    prompt = q["prompt_original"]
    complexidade = analisar_complexidade_prompt(prompt)
    linha = {
        "id": f"{args.prefixo}_{q['id']}",
        "area": q.get("tarefa", "") or args.fonte,
        "fonte": args.fonte,
        "executor": args.executor,
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_experimento": VERSAO_EXPERIMENTO,
        "gabarito": "ABERTA",
        "enunciado": prompt,
        "qtd_palavras": complexidade["metricas"]["quantidade_palavras"],
        "qtd_caracteres": complexidade["metricas"]["quantidade_caracteres"],
        "complexidade_pontuacao": complexidade["pontuacao"],
        "complexidade_nivel": complexidade["nivel"],
        "conversation_hash": q.get("conversation_hash", ""),
        "grupo": q.get("grupo", ""),
        "tipo_correcao": "juiz",
        "regra_rotulo": REGRA_ROTULO,
        "finish_reason_A": a["finish_reason"],
        "finish_reason_B": b["finish_reason"],
    }
    for m in METRICAS_CONFIG:
        linha[f"{m}_A"] = a[m]
        linha[f"{m}_B"] = b[m]
    if a["custo_usd"] is not None and b["custo_usd"] is not None:
        linha["economia_usd"] = b["custo_usd"] - a["custo_usd"]
        linha["pct_economia_custo"] = (round((b["custo_usd"] - a["custo_usd"]) / b["custo_usd"] * 100, 2)
                                       if b["custo_usd"] else None)
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
    faltando = [c for c in ("id", "prompt_original") if c not in df.columns]
    if faltando:
        sys.exit(f"Erro: colunas ausentes no CSV de entrada: {faltando}")
    df = df[df["prompt_original"].str.strip() != ""]
    if df["id"].duplicated().any():
        sys.exit("Erro: há ids repetidos no CSV de entrada.")
    return df


# ---------------------------------------------------------------------------
# Modo 1: coletar
# ---------------------------------------------------------------------------
def coletar(args, chave_mistral):
    entrada, saida = Path(args.entrada), Path(args.saida)
    if not entrada.exists():
        sys.exit(f"Erro: arquivo de entrada não encontrado: {entrada}")
    prompts = carregar_prompts(entrada)

    if not args.sem_ecologits:
        from ecologits import EcoLogits
        EcoLogits.init(providers=["google_genai"])
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    feitos = ids_ja_processados(saida)
    pendentes = prompts[~prompts["id"].map(lambda i: f"{args.prefixo}_{i}").isin(feitos)]
    meta = min(args.limite, len(pendentes)) if args.limite and args.limite > 0 else len(pendentes)

    print(f"Executor: {args.executor} | Fonte: {args.fonte} | A: {CONFIGURACOES['A']['modelo']} | "
          f"B: {CONFIGURACOES['B']['modelo']} | "
          f"Juiz: {'DESLIGADO (--sem-juiz)' if args.sem_juiz else args.modelo_juiz}")
    print(f"Prompts: {len(prompts)} | já feitos: {len(feitos)} | pendentes: {len(pendentes)} | "
          f"meta desta execução: {meta}")

    novo = not saida.exists() or saida.stat().st_size == 0
    contagem, gravados, falhas = {}, 0, 0
    with open(saida, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS, extrasaction="ignore")
        if novo:
            writer.writeheader()
            f.flush()
        try:
            for _, q in pendentes.iterrows():
                if gravados >= meta:
                    break
                print(f"[{gravados + 1}/{meta}] {args.prefixo}_{q['id']} "
                      f"({len(q['prompt_original'])} caracteres)")
                try:
                    a = chamar_config(client, CONFIGURACOES["A"], q["prompt_original"])
                    time.sleep(args.pausa)
                    b = chamar_config(client, CONFIGURACOES["B"], q["prompt_original"])
                    time.sleep(args.pausa)
                except RuntimeError as e:
                    print(f"    x não gravado: {e}", file=sys.stderr)
                    falhas += 1
                    continue

                linha = montar_linha(q, args, a, b)
                if args.sem_juiz:
                    veredito = {"juiz_modelo": args.modelo_juiz, "juiz_status": "pendente"}
                else:
                    veredito = julgar(chave_mistral, args.modelo_juiz, q["prompt_original"],
                                      a["resposta_texto"], b["resposta_texto"])
                aplicar_veredito(linha, veredito)

                writer.writerow(linha)
                f.flush()
                os.fsync(f.fileno())
                gravados += 1
                contagem[linha["rotulo"]] = contagem.get(linha["rotulo"], 0) + 1
                print(f"    A atende={linha['acertou_A']} | B atende={linha['acertou_B']} "
                      f"-> {linha['rotulo']}")
        except KeyboardInterrupt:
            print("\nInterrompido. Rode o mesmo comando para continuar de onde parou.")
        except CotaDiariaEsgotada as e:
            print("\n!!! COTA DIÁRIA DO GEMINI ESGOTADA - coleta encerrada com segurança.", file=sys.stderr)
            print(f"    {str(e)[:300]}", file=sys.stderr)
            print("    A linha em curso NÃO foi gravada. Rode o mesmo comando amanhã.", file=sys.stderr)

    print(f"\nConcluído. Gravados: {gravados} | Falhas: {falhas}. Saída: {saida}")
    print("Rótulos nesta execução:")
    for rotulo, qtd in sorted(contagem.items()):
        print(f"  {rotulo}: {qtd}")


# ---------------------------------------------------------------------------
# Modo 2: julgar as linhas pendentes de um arquivo já coletado
# ---------------------------------------------------------------------------
def julgar_pendentes(args, chave_mistral):
    saida = Path(args.saida)
    if not saida.exists():
        sys.exit(f"Erro: arquivo não encontrado: {saida}")
    df = pd.read_csv(saida, dtype=str, keep_default_na=False)
    for col in COLUNAS:
        if col not in df.columns:
            df[col] = ""
    pendentes = df.index[df["juiz_status"].isin(["pendente", "erro", ""])].tolist()
    if args.limite and args.limite > 0:
        pendentes = pendentes[: args.limite]
    print(f"Linhas no arquivo: {len(df)} | a julgar: {len(pendentes)} | juiz: {args.modelo_juiz}")

    contagem = {}
    try:
        for n, i in enumerate(pendentes, 1):
            linha = df.loc[i].to_dict()
            print(f"[{n}/{len(pendentes)}] {linha['id']}")
            veredito = julgar(chave_mistral, args.modelo_juiz, linha["enunciado"],
                              linha["resposta_texto_A"], linha["resposta_texto_B"])
            linha = aplicar_veredito(linha, veredito)
            for col, valor in linha.items():
                df.at[i, col] = "" if valor is None else str(valor)
            temporario = saida.with_suffix(".tmp")
            df[COLUNAS].to_csv(temporario, index=False)
            temporario.replace(saida)
            contagem[linha["rotulo"]] = contagem.get(linha["rotulo"], 0) + 1
            print(f"    A atende={linha['acertou_A']} | B atende={linha['acertou_B']} -> {linha['rotulo']}")
            time.sleep(args.pausa)
    except KeyboardInterrupt:
        print("\nInterrompido. Rode o mesmo comando para continuar.")
    print("Rótulos nesta execução:")
    for rotulo, qtd in sorted(contagem.items()):
        print(f"  {rotulo}: {qtd}")


def main():
    ap = argparse.ArgumentParser(description="Coletor TokenWise para prompts abertos (A x B + juiz).")
    ap.add_argument("--entrada", help="CSV com id e prompt_original (ex.: wildchat_amostra_50.csv)")
    ap.add_argument("--saida", default=None, help="padrão: resultado_aberto_<executor>.csv")
    ap.add_argument("--executor", default=None, help="nome ou apelido de quem roda (não a chave!)")
    ap.add_argument("--fonte", default="wildchat", help="nome da fonte gravado em cada linha")
    ap.add_argument("--prefixo", default=None, help="prefixo dos ids (padrão: o nome da fonte)")
    ap.add_argument("--limite", type=int, default=0, help="quantas linhas processar nesta execução")
    ap.add_argument("--pausa", type=float, default=None,
                    help="segundos entre chamadas (padrão: SLEEP_ENTRE_CHAMADAS ou 4.5)")
    ap.add_argument("--modelo-juiz", default=None,
                    help=f"modelo da Mistral usado como juiz (padrão: MODELO_JUIZ ou {MODELO_JUIZ_PADRAO})")
    ap.add_argument("--sem-juiz", action="store_true", help="coleta só A e B; julga depois")
    ap.add_argument("--julgar-pendentes", action="store_true",
                    help="não coleta: julga as linhas pendentes do arquivo --saida")
    ap.add_argument("--sem-ecologits", action="store_true", help="não estimar energia/água/carbono")
    args = ap.parse_args()

    load_dotenv(PASTA / ".env")
    args.pausa = args.pausa if args.pausa is not None else float(os.getenv("SLEEP_ENTRE_CHAMADAS", "4.5"))
    args.modelo_juiz = args.modelo_juiz or os.getenv("MODELO_JUIZ", MODELO_JUIZ_PADRAO)
    args.prefixo = args.prefixo or args.fonte

    precisa_juiz = args.julgar_pendentes or not args.sem_juiz
    chave_mistral = os.getenv("MISTRAL_API_KEY")
    if precisa_juiz and not chave_mistral:
        sys.exit("Erro: defina MISTRAL_API_KEY no arquivo .env (ou use --sem-juiz).")

    if args.julgar_pendentes:
        if not args.saida:
            sys.exit("Erro: informe --saida com o arquivo a julgar.")
        julgar_pendentes(args, chave_mistral)
        return

    if not args.entrada or not args.executor:
        sys.exit("Erro: informe --entrada e --executor.")
    gemini = os.getenv("GEMINI_API_KEY")
    if not gemini or gemini == "cole_sua_chave_aqui":
        sys.exit("Erro: defina GEMINI_API_KEY no arquivo .env.")
    args.saida = args.saida or f"resultado_aberto_{args.executor}.csv"
    coletar(args, chave_mistral)


if __name__ == "__main__":
    main()
