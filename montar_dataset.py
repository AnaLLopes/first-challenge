#!/usr/bin/env python3
"""TokenWise - monta o dataset final (uma linha por prompt, em 6 blocos) para o treino do ML.

Junta os arquivos de coleta de todo o squad e de todas as fontes:
  - coletar_piloto.py  -> resultado_<apelido>.csv          (ENEM, corrigido por gabarito)
  - coletar_aberto.py  -> resultado_aberto_<apelido>.csv   (WildChat, corrigido por juiz)

Uso:
    python montar_dataset.py resultado_*.csv
    python montar_dataset.py pasta_com_resultados/ --saida dataset_final.csv

O que ele faz:
  1. lê todos os arquivos e organiza as colunas nos 6 blocos do formato final;
  2. remove repetições do mesmo prompt (ex.: as questões comuns do ENEM rodadas por todos),
     mantendo a primeira coleta e marcando as outras como fora do treino;
  3. calcula alvo (1 = A basta, 0 = precisa de B) e usar_no_treino;
  4. divide em treino / validação / teste (70/15/15), mantendo cada `grupo` inteiro do mesmo
     lado e a proporção de alvo parecida nos três conjuntos.

Requisitos: pip install pandas
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import pandas as pd

# Licença de cada fonte: CONFIRMAR antes da apresentação
LICENCAS = {
    "wildchat": "WildChat (Allen AI) - confirmar licença na página do dataset",
    "dolly-15k": "Databricks Dolly 15k - CC BY-SA 3.0 (confirmar)",
}
LICENCA_ENEM = "Provas do ENEM/INEP - confirmar termos de uso"

PROPORCOES = {"treino": 0.70, "validacao": 0.15, "teste": 0.15}

BLOCOS = {
    "1_identificacao": ["id", "fonte", "licenca", "grupo", "executor", "coletado_em",
                        "versao_experimento"],
    "2_entrada_ml": ["prompt", "tokens_entrada", "qtd_palavras", "qtd_caracteres",
                     "complexidade_pontuacao", "complexidade_nivel", "tipo_tarefa"],
    "3_referencia": ["tipo_correcao", "resposta_referencia", "juiz_modelo", "juiz_justificativa"],
    "4_modelo_A": [], "5_modelo_B": [],
    "6_alvo": ["rotulo", "alvo", "usar_no_treino", "observacao", "split"],
}
METRICAS = ["modelo", "resposta", "atendeu", "tokens_saida", "tokens_raciocinio", "custo_usd",
            "latencia_s", "agua_l", "energia_kwh", "carbono_kg", "status", "finish_reason"]
BLOCOS["4_modelo_A"] = [f"{m}_A" for m in METRICAS]
BLOCOS["5_modelo_B"] = [f"{m}_B" for m in METRICAS]
COLUNAS_FINAIS = [c for cols in BLOCOS.values() for c in cols]


def listar_arquivos(entradas):
    arquivos = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            arquivos += sorted(p.glob("resultado*.csv"))
        else:
            arquivos += [Path(x) for x in sorted(glob.glob(e))]
    if not arquivos:
        sys.exit("Erro: nenhum arquivo de resultado encontrado.")
    return arquivos


def padronizar(df, arquivo):
    """Converte um arquivo de coleta (ENEM ou aberto) para as colunas finais."""
    col = lambda nome: df[nome] if nome in df else pd.Series([None] * len(df), index=df.index)
    eh_aberto = "tipo_correcao" in df and (df["tipo_correcao"] == "juiz").any()
    fonte = col("fonte").fillna("").astype(str)
    eh_enem = fonte.str.contains("ENEM", case=False)

    out = pd.DataFrame(index=df.index)
    # Bloco 1
    out["id"] = col("id")
    out["fonte"] = fonte.where(~eh_enem, "enem2026")
    out["licenca"] = out["fonte"].map(LICENCAS).where(~eh_enem, LICENCA_ENEM)
    out["grupo"] = col("grupo").where(col("grupo").fillna("") != "", col("id"))
    out["executor"] = col("executor")
    out["coletado_em"] = col("coletado_em_utc")
    out["versao_experimento"] = col("versao_experimento")
    # Bloco 2
    out["prompt"] = col("enunciado")
    out["tokens_entrada"] = col("tokens_entrada_A")
    out["qtd_palavras"] = col("qtd_palavras")
    out["qtd_caracteres"] = col("qtd_caracteres")
    out["complexidade_pontuacao"] = col("complexidade_pontuacao")
    out["complexidade_nivel"] = col("complexidade_nivel")
    out["tipo_tarefa"] = col("area")
    # ENEM: completa a área pelo número da questão quando ela veio vazia
    numero = pd.to_numeric(col("id").astype(str).str.extract(r"Q(\d+)$")[0], errors="coerce")
    area_enem = pd.cut(numero, [0, 5, 45, 90, 135, 180],
                       labels=["lingua_estrangeira", "linguagens", "humanas", "natureza", "matematica"])
    sem_area = eh_enem & out["tipo_tarefa"].isna()
    out.loc[sem_area, "tipo_tarefa"] = area_enem[sem_area].astype(str)
    # Bloco 3
    out["tipo_correcao"] = "juiz" if eh_aberto else "gabarito"
    out["resposta_referencia"] = None if eh_aberto else col("gabarito")
    out["juiz_modelo"] = col("juiz_modelo")
    out["juiz_justificativa"] = col("juiz_justificativa")
    # Blocos 4 e 5
    for s in ("A", "B"):
        out[f"modelo_{s}"] = col(f"modelo_{s}")
        out[f"resposta_{s}"] = col(f"resposta_texto_{s}")
        out[f"atendeu_{s}"] = col(f"acertou_{s}")
        for m in ("tokens_saida", "tokens_raciocinio", "custo_usd", "latencia_s",
                  "agua_l", "energia_kwh", "carbono_kg", "status"):
            out[f"{m}_{s}"] = col(f"{m}_{s}")
        out[f"finish_reason_{s}"] = col(f"finish_reason_{s}")
    # Bloco 6
    out["rotulo"] = col("rotulo")
    out["arquivo_origem"] = arquivo.name
    return out


def dividir(df, semente):
    """Divide os grupos em treino/validação/teste, estratificando pelo alvo predominante."""
    rng = random.Random(semente)
    df["split"] = ""
    treino = df[df["usar_no_treino"]]
    resumo = treino.groupby("grupo").agg(n=("alvo", "size"), alvo=("alvo", lambda s: int(s.mean() >= 0.5)))
    for _, estrato in resumo.groupby("alvo"):
        grupos = list(estrato.index)
        rng.shuffle(grupos)
        total, acumulado = estrato["n"].sum(), 0
        for g in grupos:
            fracao = acumulado / total
            split = ("treino" if fracao < PROPORCOES["treino"] else
                     "validacao" if fracao < PROPORCOES["treino"] + PROPORCOES["validacao"] else "teste")
            df.loc[(df["grupo"] == g) & df["usar_no_treino"], "split"] = split
            acumulado += estrato.loc[g, "n"]
    return df


def main():
    ap = argparse.ArgumentParser(description="Monta o dataset final do TokenWise.")
    ap.add_argument("entradas", nargs="+", help="arquivos resultado_*.csv ou uma pasta")
    ap.add_argument("--saida", default="dataset_final.csv")
    ap.add_argument("--semente", type=int, default=2026)
    args = ap.parse_args()

    arquivos = listar_arquivos(args.entradas)
    partes = []
    for arq in arquivos:
        df = pd.read_csv(arq, dtype=str, keep_default_na=False).replace({"": None})
        partes.append(padronizar(df, arq))
        print(f"  {arq.name}: {len(df)} linhas")
    dados = pd.concat(partes, ignore_index=True)

    # Repetições: mesmo id (ex.: questões comuns do ENEM rodadas por todos) -> fica a primeira
    dados = dados.sort_values("coletado_em", kind="stable")
    repetida = dados.duplicated(subset="id", keep="first")
    dados["observacao"] = ""
    dados.loc[repetida, "observacao"] = "repetição do mesmo prompt (mantida só a primeira coleta)"

    # Alvo e uso no treino
    dados["alvo"] = dados["rotulo"].map({"economica_suficiente": 1, "referencia_necessaria": 0})
    dados["usar_no_treino"] = dados["alvo"].notna() & ~repetida
    dados.loc[dados["alvo"].isna() & ~repetida, "observacao"] = (
        "fora do treino: rótulo " + dados["rotulo"].fillna("ausente"))
    dados["alvo"] = dados["alvo"].astype("Int64")

    dados = dividir(dados, args.semente)
    final = dados[COLUNAS_FINAIS + ["arquivo_origem"]].sort_values(["fonte", "id"])
    final.to_csv(args.saida, index=False, encoding="utf-8")

    # Resumo
    usados = final[final["usar_no_treino"]]
    print(f"\nLinhas no total: {len(final)} | usadas no treino: {len(usados)}")
    print("\nRótulos por fonte:")
    print(pd.crosstab(final["fonte"], final["rotulo"].fillna("(sem rótulo)"), margins=True,
                      margins_name="total").to_string())
    if len(usados):
        pct = (usados["alvo"] == 0).mean() * 100
        print(f"\nClasse menos comum ('referencia_necessaria', alvo = 0): {pct:.1f}% dos usados no treino")
        print("\nDivisão (alvo 1 = A basta, 0 = precisa de B):")
        print(pd.crosstab(usados["split"], usados["alvo"], margins=True, margins_name="total").to_string())
    print(f"\nDataset final salvo em {args.saida}")


if __name__ == "__main__":
    main()