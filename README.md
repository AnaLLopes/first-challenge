# TokenWise (MVP)

Ferramenta para otimizar prompts de IA com foco em **eficiência de tokens, redução de custo e sustentabilidade**.

Este MVP coleta o dataset próprio e treina uma primeira versão de um modelo de ML que estima o potencial de economia de tokens e o risco de perda de qualidade ao otimizar um prompt. A redução de tokens reduz proporcionalmente o custo, a latência e a pegada ambiental estimada da inferência:

1. envia os dois ao Google Gemini;
2. mede latência e tokens de entrada/saída;
3. estima energia (kWh), água (L) e carbono (kg CO₂eq) com o [EcoLogits](https://ecologits.ai);
4. calcula % de redução de tokens e a economia de recursos;
5. salva tudo em CSV, linha a linha.

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

O `.env` já está no `.gitignore`. Nunca versione a chave.

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
