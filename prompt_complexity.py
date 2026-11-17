"""Heurística local e ajustável para estimar a complexidade de prompts."""

import re
import unicodedata
from typing import Any


# Os pesos somam 100. Contagens estruturais são normalizadas e limitadas
# pelas escalas abaixo; indicadores semânticos adicionam seu peso quando ativos.
PESOS_COMPLEXIDADE = {
    "quantidade_palavras": 6,
    "quantidade_caracteres": 2,
    "quantidade_frases": 3,
    "quantidade_perguntas": 3,
    "quantidade_instrucoes": 12,
    "quantidade_requisitos": 14,
    "quantidade_etapas": 12,
    "quantidade_objetivos": 10,
    "raciocinio_logico": 8,
    "calculo_matematico": 8,
    "conhecimento_tecnico": 7,
    "multiplas_etapas": 5,
    "regras_e_restricoes": 5,
    "analise_dados": 5,
}

ESCALAS_CONTAGEM = {
    "quantidade_palavras": 100,
    "quantidade_caracteres": 600,
    "quantidade_frases": 6,
    "quantidade_perguntas": 3,
    "quantidade_instrucoes": 6,
    "quantidade_requisitos": 5,
    "quantidade_etapas": 5,
    "quantidade_objetivos": 4,
}

# Limites superiores configuráveis: 0-20 baixa, 21-39 moderada,
# 40-69 alta e 70-100 muito_alta.
LIMITES_NIVEIS = {
    "baixa": 20,
    "moderada": 39,
    "alta": 69,
}

PADROES_INSTRUCOES = re.compile(
    r"\b(?:explique|descreva|crie|escreva|implemente|desenvolva|"
    r"construa|adicione|inclua|forneca|gere|calcule|resolva|compare|"
    r"analise|avalie|identifique|liste|resuma|traduza|classifique|"
    r"extraia|mostre|apresente|organize|planeje|considere|verifique|"
    r"valide|informe|retorne|converta|ordene|filtre|recomende|"
    r"justifique|prove|demonstre|otimize|corrija|leia)\b"
)

PADROES_REQUISITOS = re.compile(
    r"\b(?:deve|devem|precisa|precisam|obrigatoriamente|apenas|somente|"
    r"sem|nao|no maximo|pelo menos|limite|restricao|restricoes|"
    r"formato|entre|exatamente|incluindo|excluindo|condicao|"
    r"autenticacao|autorizacao|validacao|seguranca|documentacao|"
    r"requisito|requisitos|tratamento de erros)\b"
)

PADROES_TRANSICAO = re.compile(
    r"\b(?:primeiro|primeira|depois|em seguida|posteriormente|"
    r"por fim|finalmente|entao)\b"
)

PADROES_PASSOS = re.compile(
    r"\b(?:passo|etapa)\s*(?:numero\s*)?\d+\b"
)

PADROES_LOGICA = re.compile(
    r"\b(?:justifique|prove|demonstre|deduza|inferir|infira|"
    r"raciocine|avalie|compare|otimize|causa|consequencia|"
    r"passo a passo|por que|se .* entao)\b"
)

PADROES_CALCULO = re.compile(
    r"(?:\d\s*(?:[+\-*/×÷=]|mais|menos|vezes|dividido por)\s*\d)"
    r"|\b(?:calcule|calcular|resolva a equacao|equacao|porcentagem|"
    r"raiz quadrada|multiplique|divida|som(a|e))\b"
)

PADROES_TECNICOS = re.compile(
    r"\b(?:python|sql|api|rest|http|autenticacao|validacao|"
    r"banco de dados|algoritmo|funcao|classe|codigo|"
    r"servidor|endpoint|framework|arquitetura|llm|modelo de linguagem|"
    r"criptografia|regex|machine learning)\b"
)

PADROES_DADOS = re.compile(
    r"\b(?:analise|compare|classifique|extraia|filtre|"
    r"transforme|dados|planilha|tabela|documento|dataset|"
    r"texto fornecido|lista fornecida|a partir dos)\b"
)

GRUPOS_OBJETIVOS = {
    "explicar": re.compile(r"\b(?:explique|descreva|informe|defina)\b"),
    "criar_implementar": re.compile(
        r"\b(?:crie|escreva|implemente|desenvolva|construa|gere|adicione)\b"
    ),
    "analisar_comparar": re.compile(
        r"\b(?:analise|avalie|compare|investigue|identifique)\b"
    ),
    "transformar_conteudo": re.compile(
        r"\b(?:traduza|resuma|converta|reescreva|extraia|filtre|ordene)\b"
    ),
    "calcular_resolver": re.compile(
        r"\b(?:calcule|resolva|multiplique|divida|some|subtraia|prove)\b"
    ),
    "classificar": re.compile(r"\b(?:classifique|categorize)\b"),
    "planejar": re.compile(r"\b(?:planeje|organize|cronograma)\b"),
    "validar_testar": re.compile(
        r"\b(?:valide|verifique|teste|corrija|garanta)\b"
    ),
    "documentar": re.compile(r"\b(?:documente|documentar)\b"),
    "formatar_apresentar": re.compile(
        r"\b(?:liste|apresente|mostre|retorne|inclua|forneca)\b"
    ),
}


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(
        caractere for caractere in texto
        if not unicodedata.combining(caractere)
    )


def _pontuar(metricas: dict[str, int], indicadores: dict[str, bool]) -> int:
    """Calcula score 0-100 separadamente da classificação por nível."""
    pontos = 0.0
    for metrica, escala in ESCALAS_CONTAGEM.items():
        proporcao = min(metricas[metrica] / escala, 1.0)
        pontos += proporcao * PESOS_COMPLEXIDADE[metrica]

    for indicador, ativo in indicadores.items():
        if ativo:
            pontos += PESOS_COMPLEXIDADE[indicador]

    return round(min(pontos, 100.0))


def _classificar_nivel(pontuacao: int) -> str:
    """Converte a pontuação no nível configurado."""
    for nivel, limite in LIMITES_NIVEIS.items():
        if pontuacao <= limite:
            return nivel
    return "muito_alta"


def analisar_complexidade_prompt(prompt: str) -> dict[str, Any]:
    """Extrai métricas locais e estima a complexidade sem chamar uma LLM.

    É uma heurística textual configurável, não uma avaliação semântica
    definitiva. Comprimento tem peso baixo; requisitos, etapas e sinais de
    raciocínio/tarefa contribuem de forma independente.
    """
    texto = _normalizar(prompt)
    palavras = re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", texto)
    frases = [
        parte for parte in re.split(r"(?<=[.!?])\s+|\n+", texto.strip())
        if parte.strip()
    ]
    quantidade_perguntas = texto.count("?")
    quantidade_instrucoes = len(PADROES_INSTRUCOES.findall(texto))
    quantidade_requisitos = len(PADROES_REQUISITOS.findall(texto))

    marcadores_numerados = len(re.findall(r"(?m)^\s*\d+[.)]\s+", texto))
    marcadores_passos = len(PADROES_PASSOS.findall(texto))
    transicoes = len(PADROES_TRANSICAO.findall(texto))
    quantidade_etapas = max(
        marcadores_numerados,
        marcadores_passos,
        transicoes + 1 if transicoes else 0,
        min(quantidade_instrucoes, 5),
    )

    quantidade_objetivos = sum(
        1 for padrao in GRUPOS_OBJETIVOS.values()
        if padrao.search(texto)
    )

    indicadores = {
        "raciocinio_logico": bool(PADROES_LOGICA.search(texto)),
        "calculo_matematico": bool(PADROES_CALCULO.search(texto)),
        "conhecimento_tecnico": bool(PADROES_TECNICOS.search(texto)),
        "multiplas_etapas": quantidade_etapas > 1,
        "regras_e_restricoes": (
            quantidade_requisitos > 0
            or bool(re.search(r"\b(?:se|caso|desde que|condicao)\b", texto))
        ),
        "analise_dados": bool(PADROES_DADOS.search(texto)),
    }

    metricas = {
        "quantidade_palavras": len(palavras),
        "quantidade_caracteres": len(prompt),
        "quantidade_frases": len(frases),
        "quantidade_perguntas": quantidade_perguntas,
        "quantidade_instrucoes": quantidade_instrucoes,
        "quantidade_requisitos": quantidade_requisitos,
        "quantidade_etapas": quantidade_etapas,
        "quantidade_objetivos": quantidade_objetivos,
    }
    pontuacao = _pontuar(metricas, indicadores)

    return {
        "pontuacao": pontuacao,
        "nivel": _classificar_nivel(pontuacao),
        "metodo": "heuristico",
        "metricas": metricas,
        "indicadores": indicadores,
        "status_analise": "sucesso",
    }
