#!/usr/bin/env python3
"""TokenWise - ferramentas para os arquivos .parquet do WildChat.

Três comandos, usados nesta ordem:

  1. inspecionar  -> quantas conversas há e em quais idiomas (lê só a coluna de idioma)
        python wildchat.py inspecionar PASTA_DOS_PARQUETS

  2. extrair      -> tira o PRIMEIRO pedido do usuário de cada conversa no idioma escolhido,
                     remove tóxicos e repetidos, e mostra o tamanho dos prompts
        python wildchat.py extrair PASTA_DOS_PARQUETS --idioma Portuguese --saida wildchat_pt.csv

  3. amostra      -> filtra os prompts que não são autocontidos, agrupa os parecidos e
                     sorteia N prompts equilibrando as faixas de tamanho, no formato que o
                     coletor lê (id, tarefa, prompt_original, grupo)
        python wildchat.py amostra wildchat_pt.csv --n 50 --saida wildchat_amostra_50.csv

     Os prompts removidos pelo filtro são gravados em <saida>_excluidos.csv, com o motivo,
     para conferência.

Não copia dados pessoais (IP, país, estado, navegador). Lê os arquivos aos poucos,
então funciona com arquivos de centenas de MB.

Requisitos: pip install pandas pyarrow
"""

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

FAIXAS = [0, 50, 200, 1000, 4000, float("inf")]
ROTULOS_FAIXAS = ["até 50", "51-200", "201-1000", "1001-4000", "acima de 4000"]


def fmt(n):
    return f"{n:,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Filtro de prompts autocontidos e agrupamento
# ---------------------------------------------------------------------------
# Um prompt "autocontido" tem tudo o que é preciso para ser respondido no próprio texto.
# As regras abaixo são simples e conservadoras; os removidos ficam num arquivo para conferência.
PADRAO_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)
PADROES_SEM_TAREFA = [
    # pedidos que dependem de mensagens futuras
    (re.compile(r"\b(irei|vou|vamos) (te |lhe )?(mandar|enviar|passar)\b", re.IGNORECASE),
     "depende de mensagens seguintes"),
    (re.compile(r"\b(te |lhe )?(mandarei|enviarei)\b", re.IGNORECASE),
     "depende de mensagens seguintes"),
    (re.compile(r"\baguarde (as |minhas )?(próximas|proximas|instruções|instrucoes)\b", re.IGNORECASE),
     "depende de mensagens seguintes"),
    # perguntas sobre o próprio chatbot
    (re.compile(r"\bvoc[eê] (consegue |pode )?(lembrar|se lembra|lembra)\b", re.IGNORECASE),
     "pergunta sobre o próprio chatbot"),
    (re.compile(r"\b(qual (é )?(o )?seu nome|quem (é você|te criou)|você é o chatgpt)\b", re.IGNORECASE),
     "pergunta sobre o próprio chatbot"),
]


def motivo_exclusao(prompt, min_caracteres, max_caracteres):
    """Devolve o motivo para excluir o prompt, ou None se ele pode ser usado."""
    tamanho = len(prompt)
    if tamanho < min_caracteres:
        return f"curto demais (< {min_caracteres} caracteres)"
    if tamanho > max_caracteres:
        return f"longo demais (> {max_caracteres} caracteres)"
    if PADRAO_LINK.search(prompt):
        return "depende de link (imagem, site ou arquivo)"
    for padrao, motivo in PADROES_SEM_TAREFA:
        if padrao.search(prompt):
            return motivo
    return None


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", texto)


def chave_grupo(prompt, n_palavras=3):
    """Agrupa prompts que começam com as mesmas palavras (ex.: vários "parafraseie várias vezes ...").

    É uma regra simples: serve para manter pedidos do mesmo modelo juntos na divisão
    treino/teste. Pode ser refinada depois (ex.: por similaridade de texto).
    """
    palavras = normalizar(prompt)[:n_palavras]
    return "_".join(palavras) if palavras else "vazio"


def listar_parquets(pasta):
    pasta = Path(pasta)
    if not pasta.exists():
        sys.exit(f"Erro: não encontrei {pasta}")
    if pasta.is_file():
        return [pasta]
    arquivos = sorted(pasta.rglob("*.parquet"))
    if not arquivos:
        sys.exit(f"Erro: nenhum arquivo .parquet em {pasta}")
    return arquivos


def primeira_mensagem_usuario(conversa):
    for msg in conversa if conversa is not None else []:
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# 1. inspecionar
# ---------------------------------------------------------------------------
def inspecionar(args):
    arquivos = listar_parquets(args.pasta)
    print(f"{len(arquivos)} arquivo(s) .parquet")
    primeiro = pq.ParquetFile(arquivos[0])
    print("\nColunas:")
    for campo in primeiro.schema_arrow:
        tipo = str(campo.type)
        print(f"  {campo.name}: {tipo[:70]}{'...' if len(tipo) > 70 else ''}")

    idiomas, total = Counter(), 0
    for n, caminho in enumerate(arquivos, 1):
        arq = pq.ParquetFile(caminho)
        print(f"[{n}/{len(arquivos)}] {caminho.name}: {fmt(arq.metadata.num_rows)} conversas")
        for lote in arq.iter_batches(batch_size=20000, columns=["language"]):
            valores = lote.column("language").to_pylist()
            idiomas.update(v or "?" for v in valores)
            total += len(valores)

    print(f"\nTotal de conversas: {fmt(total)}")
    print("Idiomas (coluna 'language' do dataset):")
    for idioma, qtd in idiomas.most_common(args.top):
        print(f"  {idioma:<15} {fmt(qtd):>10}  ({qtd / total * 100:.2f}%)")
    print(f"\nPortuguese: {fmt(idiomas.get('Portuguese', 0))}")


# ---------------------------------------------------------------------------
# 2. extrair
# ---------------------------------------------------------------------------
def extrair(args):
    arquivos = listar_parquets(args.pasta)
    colunas = ["conversation_hash", "timestamp", "language", "turn", "toxic", "conversation"]

    registros, lidas, no_idioma, toxicas, vazias = [], 0, 0, 0, 0
    for n, caminho in enumerate(arquivos, 1):
        print(f"[{n}/{len(arquivos)}] {caminho.name}")
        for lote in pq.ParquetFile(caminho).iter_batches(batch_size=5000, columns=colunas):
            df = lote.to_pandas()
            lidas += len(df)
            df = df[df["language"] == args.idioma]
            no_idioma += len(df)
            toxica = df["toxic"].fillna(False).astype(bool)
            toxicas += int(toxica.sum())
            for _, linha in df[~toxica].iterrows():
                prompt = primeira_mensagem_usuario(linha["conversation"])
                if not prompt:
                    vazias += 1
                    continue
                registros.append({
                    "conversation_hash": linha["conversation_hash"],
                    "timestamp": linha["timestamp"],
                    "turnos": linha["turn"],
                    "prompt": prompt,
                    "qtd_caracteres": len(prompt),
                })

    saida = pd.DataFrame(registros)
    antes = len(saida)
    if antes:
        saida = saida.drop_duplicates(subset="prompt").reset_index(drop=True)
    repetidos = antes - len(saida)
    saida.to_csv(args.saida, index=False)

    print(f"\nConversas lidas: {fmt(lidas)}")
    print(f"No idioma '{args.idioma}': {fmt(no_idioma)}")
    print(f"  descartadas por toxicidade: {fmt(toxicas)}")
    print(f"  sem pedido do usuário: {fmt(vazias)}")
    print(f"  prompts repetidos removidos: {fmt(repetidos)}")
    print(f"  prompts gravados em {args.saida}: {fmt(len(saida))}")
    if not saida.empty:
        resumo_tamanho(saida)


def resumo_tamanho(df):
    tam = df["qtd_caracteres"]
    print("\nTamanho dos prompts (caracteres):")
    print(f"  mediana {tam.median():.0f} | média {tam.mean():.0f} | 25% {tam.quantile(.25):.0f} | "
          f"75% {tam.quantile(.75):.0f} | 90% {tam.quantile(.9):.0f} | máximo {tam.max():.0f}")
    faixas = pd.cut(tam, FAIXAS, labels=ROTULOS_FAIXAS)
    print("  por faixa:")
    for faixa, qtd in faixas.value_counts().reindex(ROTULOS_FAIXAS).items():
        print(f"    {faixa:>14}: {fmt(qtd):>8} ({qtd / len(df) * 100:.1f}%)")
    if "turnos" in df:
        print(f"  conversas com mais de um turno: {(df['turnos'] > 1).mean() * 100:.1f}% "
              "(só o primeiro pedido foi extraído)")


# ---------------------------------------------------------------------------
# 3. amostra
# ---------------------------------------------------------------------------
def amostra(args):
    df = pd.read_csv(args.csv)
    if "prompt" not in df:
        sys.exit("Erro: o CSV precisa da coluna 'prompt' (use o arquivo gerado por 'extrair').")
    df["prompt"] = df["prompt"].astype(str)
    df["qtd_caracteres"] = df["prompt"].str.len()

    # Filtro: só prompts autocontidos e dentro dos limites de tamanho
    df["motivo_exclusao"] = df["prompt"].map(
        lambda p: motivo_exclusao(p, args.min_caracteres, args.max_caracteres))
    excluidos = df[df["motivo_exclusao"].notna()]
    df = df[df["motivo_exclusao"].isna()].copy()
    arquivo_excluidos = Path(args.saida).with_name(Path(args.saida).stem + "_excluidos.csv")
    excluidos[["conversation_hash", "motivo_exclusao", "qtd_caracteres", "prompt"]].to_csv(
        arquivo_excluidos, index=False)

    total = len(df) + len(excluidos)
    print(f"Prompts no arquivo: {fmt(total)}")
    print(f"Excluídos pelo filtro: {fmt(len(excluidos))} ({len(excluidos) / total * 100:.1f}%) "
          f"-> conferência em {arquivo_excluidos}")
    for motivo, qtd in excluidos["motivo_exclusao"].value_counts().items():
        print(f"  {motivo}: {fmt(qtd)}")
    print(f"Prompts disponíveis para o sorteio: {fmt(len(df))}\n")
    if df.empty:
        sys.exit("Erro: nenhum prompt passou pelo filtro.")

    # Grupo: prompts que começam com as mesmas palavras ficam juntos
    df["grupo"] = df["prompt"].map(chave_grupo)
    df["faixa"] = pd.cut(df["qtd_caracteres"], FAIXAS, labels=ROTULOS_FAIXAS)

    n = min(args.n, len(df))
    faixas = [f for f in ROTULOS_FAIXAS if (df["faixa"] == f).any()]
    cota = {f: n // len(faixas) for f in faixas}
    for f in faixas[: n - sum(cota.values())]:
        cota[f] += 1

    partes, sobra = [], 0
    for f in faixas:
        grupo = df[df["faixa"] == f]
        k = min(cota[f], len(grupo))
        sobra += cota[f] - k
        partes.append(grupo.sample(k, random_state=args.semente))
    escolhidos = pd.concat(partes)
    if sobra:  # faixas pequenas: completa com prompts das outras faixas
        resto = df.drop(escolhidos.index)
        escolhidos = pd.concat([escolhidos, resto.sample(min(sobra, len(resto)), random_state=args.semente)])
    escolhidos = escolhidos.sample(frac=1, random_state=args.semente).reset_index(drop=True)

    saida = pd.DataFrame({
        "id": range(1, len(escolhidos) + 1),
        "tarefa": "wildchat",
        "prompt_original": escolhidos["prompt"],
        "conversation_hash": escolhidos.get("conversation_hash"),
        "grupo": escolhidos["grupo"],
        "qtd_caracteres": escolhidos["qtd_caracteres"],
    })
    saida.to_csv(args.saida, index=False)
    print(f"{len(saida)} prompts sorteados (semente {args.semente}) -> {args.saida}")
    print(saida["qtd_caracteres"].pipe(pd.cut, FAIXAS, labels=ROTULOS_FAIXAS)
          .value_counts().reindex(ROTULOS_FAIXAS).loc[lambda s: s > 0]
          .rename_axis("faixa de tamanho").rename("prompts").to_string())


def main():
    ap = argparse.ArgumentParser(description="Ferramentas para os .parquet do WildChat.")
    sub = ap.add_subparsers(dest="comando", required=True)

    p1 = sub.add_parser("inspecionar", help="colunas, total de conversas e idiomas")
    p1.add_argument("pasta", help="pasta com os .parquet (ou um arquivo)")
    p1.add_argument("--top", type=int, default=15, help="quantos idiomas mostrar")
    p1.set_defaults(func=inspecionar)

    p2 = sub.add_parser("extrair", help="primeiro pedido de cada conversa num idioma")
    p2.add_argument("pasta", help="pasta com os .parquet (ou um arquivo)")
    p2.add_argument("--idioma", default="Portuguese", help="valor da coluna 'language' (padrão: Portuguese)")
    p2.add_argument("--saida", default="wildchat_pt.csv")
    p2.set_defaults(func=extrair)

    p3 = sub.add_parser("amostra", help="sorteia N prompts no formato do coletor")
    p3.add_argument("csv", help="CSV gerado pelo comando 'extrair'")
    p3.add_argument("--n", type=int, default=50)
    p3.add_argument("--min-caracteres", type=int, default=20)
    p3.add_argument("--max-caracteres", type=int, default=4000)
    p3.add_argument("--semente", type=int, default=2026, help="mesma semente = mesma amostra")
    p3.add_argument("--saida", default="wildchat_amostra.csv")
    p3.set_defaults(func=amostra)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()