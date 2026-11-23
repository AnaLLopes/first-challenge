# TokenWise (MVP)

Sistema de recomendação de configurações de IA com foco em eficiência, redução de custos e sustentabilidade.

**Objetivo**
O TokenWise analisa uma demanda antes de sua execução e recomenda, entre configurações previamente avaliadas, uma alternativa econômica com probabilidade suficiente de atender aos critérios de qualidade definidos pelo projeto.

O componente de recomendação utiliza Machine Learning tradicional, sem LLM. LLMs podem ser utilizados na coleta experimental, em tarefas auxiliares e, opcionalmente, na execução da configuração escolhida pelo usuário.

**Escopo do MVP**
A primeira versão comparará duas configurações:

Configuração A: alternativa econômica.

Configuração B: alternativa de referência.

Cada configuração será identificada pelo modelo, provedor e parâmetros utilizados. Essas definições serão registradas na documentação dos experimentos.

O problema inicial de ML será:

Dada uma demanda, prever se a configuração econômica será suficiente para produzir uma resposta aceitável.

A recomendação considerará qualidade e custo. Uma resposta mais curta ou com menos tokens não será considerada melhor se deixar de cumprir a tarefa.

**Funcionamento previsto**
O usuário informa uma demanda.

O sistema extrai características do texto.

Um classificador de ML estima a probabilidade de a configuração econômica ser suficiente.

Uma política de decisão utiliza essa previsão para recomendar uma configuração.

A interface apresenta a recomendação e suas limitações.

Opcionalmente, o usuário pode executar a demanda pela API disponível.

Quando a confiança for insuficiente, o sistema poderá recomendar a configuração de referência por precaução. Essa indicação não representa garantia de correção.

**Construção do dataset**
O dataset será produzido a partir de demandas próprias e/ou fontes públicas submetidas a refinamento da equipe.

O refinamento incluirá:

Seleção e documentação das fontes e licenças.

Limpeza, deduplicação e agrupamento de demandas semelhantes.

Padronização dos registros.

Coleta ou aproveitamento documentado de respostas das configurações avaliadas.

Avaliação da qualidade por critérios explícitos.

Construção dos rótulos utilizados no treinamento.

Separação dos conjuntos de treino, validação e teste.

Cada demanda será associada aos resultados das configurações avaliadas. Registros sem informação suficiente para estabelecer qualidade ou custo serão identificados e tratados separadamente.

**Definição do alvo**
Resultado experimental	Tratamento
A atende aos critérios e tem menor custo	Econômica suficiente
A não atende e B atende	Referência necessária
Nenhuma atende	Caso não atendido
Avaliação inconclusiva	Revisão antes do treinamento
O treinamento inicial utilizará os casos com rótulos definidos. Casos não atendidos e inconclusivos serão preservados para documentação e análise das limitações.

**Modelo de Machine Learning**
A abordagem inicial será:

Representação textual com TF-IDF.

Classificador de regressão logística.

Comparação com baselines sem aprendizado.

Somente informações disponíveis antes da execução poderão ser utilizadas como características de entrada. Respostas, consumo real de saída e avaliações de qualidade serão utilizados na construção dos rótulos e na avaliação, não como entrada da recomendação antecipada.

**Avaliação**
O modelo será comparado com:

Sempre escolher a configuração econômica.

Sempre escolher a configuração de referência.

Uma regra simples previamente definida.

As métricas incluirão:

Precisão, recall, F1 e matriz de confusão.

Taxa de recomendações econômicas inadequadas.

Taxa de respostas aceitáveis da política de recomendação.

Economia de custo em relação à configuração de referência.

Consumo de tokens observado no experimento.

Os resultados serão apresentados para o domínio, idioma e configurações efetivamente avaliados.

**Sustentabilidade**
O projeto investiga o uso mais eficiente de recursos computacionais associados à IA.

Tokens e custos serão analisados diretamente a partir dos registros disponíveis. Energia, água e emissões poderão ser estimadas com metodologia documentada, quando houver suporte para as configurações utilizadas.

Esses indicadores ambientais não representam medições diretas dos servidores. Não será assumida uma relação proporcional universal entre redução de tokens e redução de impacto ambiental.

**Estado atual da implementação**
O repositório contém um coletor experimental para comparar prompts originais e otimizados enviados ao Gemini.

Esse coletor será adaptado para executar a mesma demanda nas configurações candidatas e registrar os resultados necessários ao treinamento do recomendador.

O treinamento do classificador, a política de recomendação e a interface fazem parte das próximas entregas.

**Fora do escopo obrigatório**
Reescrita automática de prompts.

Tradução automática.

Comparação de todos os modelos disponíveis.

Aprendizado contínuo automático.

Gateway empresarial.

Medição direta de água ou energia dos provedores.

**Entrega**
Apresentação do MVP em 18 de dezembro de 2026, incluindo dataset refinado, pipeline de treinamento, modelo avaliado, interface demonstrativa e documentação das limitações.

## Configuração

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Chave gratuita do Gemini

1. Acesse [Google AI Studio](https://aistudio.google.com/apikey) e faça login com uma conta Google.
2. Clique em **Create API key** e copie a chave.
3. Copie `.env.example` para `.env` e cole a chave:

```bash
cp .env.example .env
# edite o .env: GEMINI_API_KEY=sua_chave
```

### Complexidade heurística de prompts

O coletor calcula `complexidade_prompt` localmente antes de enviar cada
solicitação ao Gemini; essa análise não faz chamadas adicionais à API. Cada
registro inclui pontuação de 0 a 100, nível, métricas estruturais, indicadores
de tarefa e `metodo: "heuristico"`. A classificação é uma estimativa textual,
não uma avaliação semântica definitiva.

Os pesos padrão ficam em `PESOS_COMPLEXIDADE` e as escalas das métricas em
`ESCALAS_CONTAGEM`, no módulo `prompt_complexity.py`. Os pesos totalizam 100:

| Critério | Peso |
|---|---:|
| Quantidade de palavras | 6 |
| Quantidade de caracteres | 2 |
| Quantidade de frases | 3 |
| Quantidade de perguntas | 3 |
| Instruções identificadas | 12 |
| Requisitos/restrições identificados | 14 |
| Etapas identificadas | 12 |
| Objetivos diferentes identificados | 10 |
| Raciocínio lógico | 8 |
| Cálculo matemático | 8 |
| Conhecimento técnico | 7 |
| Múltiplas etapas | 5 |
| Regras/restrições | 5 |
| Análise/transformação de dados | 5 |

Cada contagem contribui proporcionalmente ao seu peso, com limite definido por
sua escala (100 palavras, 600 caracteres, 6 frases, 3 perguntas, 6 instruções,
5 requisitos, 5 etapas e 4 objetivos). Indicadores booleanos adicionam o peso
completo quando detectados. O score é limitado a 100 e arredondado. Os limites
configuráveis em `LIMITES_NIVEIS` são: baixa até 20, moderada até 39, alta até
69 e muito alta acima de 69. Comprimento tem peso combinado de apenas 8 pontos,
para que o conteúdo e as exigências também influenciem a estimativa.

O `.env` já está no `.gitignore`. Nunca versione a chave.

### Exportação JSON e CSV

O [eco-gemini.py](eco-gemini.py) salva o mesmo resultado em duas formas:

- `dataset_gemini.json`: uma lista de registros completos, com objetos aninhados.
- `dataset_gemini.csv`: uma linha por registro e o nome de cada campo achatado na primeira linha.

Os campos incluem modelo, resposta, status, duração, classificação, complexidade,
tokens de entrada, saída visível, raciocínio, cache e total cobrado, além dos
valores detalhados de `usage_metadata` disponíveis no SDK. Também registra o
custo equivalente em USD, usando a tabela configurada em
`PRECOS_USD_POR_MILHAO` no próprio coletor:

- `gemini-3.5-flash-lite`: entrada US$ 0,30 / saída US$ 2,50 por milhão.
- `gemini-3.6-flash`: entrada US$ 0,75 / saída US$ 3,75 por milhão.

Os preços representam uma estimativa equivalente ao plano pago. Não significa
que uma chave gratuita foi cobrada. Tokens de raciocínio são somados aos tokens
de saída no cálculo, conforme o piloto; os detalhes podem variar conforme a
versão do SDK ou modelo retornado.

## Execução

```bash
python gerar_dataset_tokenwise.py
# ou, com arquivos e modelo específicos:
python gerar_dataset_tokenwise.py --entrada meus_prompts.csv --saida dataset_tokenwise.csv --modelo gemini-2.5-flash
```

- **Entrada**: CSV com as colunas `id, tarefa, prompt_original, prompt_otimizado` (veja `prompts_entrada_exemplo.csv`).
- **Saída**: `dataset_tokenwise.csv`, com métricas de cada versão (`*_original`, `*_otimizado`) e as colunas calculadas `pct_reducao_tokens_*`, `economia_*` e `pct_economia_*`.
- **Retomada**: cada linha é gravada assim que o par é concluído. Se o processo cair (ou você apertar Ctrl+C), rode o mesmo comando e os `id` já salvos são pulados. Pares que falharam não são gravados e são tentados de novo.
- **Limites do plano gratuito**: há uma pausa entre chamadas (`--pausa` ou `SLEEP_ENTRE_CHAMADAS`, padrão 4,5 s) e retry com backoff em caso de erro/limite de taxa.

## Observações

- Os valores do EcoLogits são **estimativas** (podem vir como intervalo; o script usa o ponto médio). Se alguma coluna ambiental vier vazia, o modelo escolhido pode não ter dados no EcoLogits ou a sua versão pode expor campos diferentes: atualize com `pip install -U ecologits`.
- Em modelos com "thinking" (ex.: Gemini 2.5), os tokens de raciocínio são somados aos tokens de saída.
- A mesma resposta pode variar entre execuções; para análises robustas, considere repetir as medições.
- `openai` está no `requirements.txt` por pedido, mas ainda não é usado pelo script.
