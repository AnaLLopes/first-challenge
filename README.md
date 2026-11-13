# first-challenge

🌿 TokenWise — Otimização de Prompts para Eficiência e Sustentabilidade em IA

O TokenWise é uma solução voltada para a otimização inteligente de prompts antes do envio para Grandes Modelos de Linguagem (LLMs). Ao remover redundâncias e instruções prolixas mantendo a qualidade e os requisitos da tarefa, o projeto visa:

Reduzir custos com consumo de tokens (API Billing).

Diminuir a latência das respostas.

Mitigar o impacto ambiental da inferência (consumo energético, pegada hídrica e emissão de carbono).

Em vez de usar uma LLM para reescrever prompts em tempo real (o que consumiria ainda mais energia), o TokenWise utiliza Machine Learning Tradicional (como Random Forest ou XGBoost) para prever o nível seguro de compressão de um prompt e estimar a economia de recursos gerada.

📁 Estrutura do Repositório

.
├── gerar_dataset_tokenwise.py   # Script automatizado para execução em lote e medição
├── prompts_entrada_exemplo.csv  # Template para preenchimento manual dos prompts pelo grupo
├── .env.example                 # Exemplo de configuração das variáveis de ambiente
├── .gitignore                   # Arquivo para ignorar .env e dados sensíveis
└── README.md                    # Documentação do projeto


🚀 Como Começar

1. Pré-requisitos

Python 3.10+ instalado.

Conta ativa e chave de API cadastrada em pelo menos um dos provedores:

OpenAI (OPENAI_API_KEY)

Anthropic / Claude (ANTHROPIC_API_KEY)

Google Gemini (GEMINI_API_KEY)

2. Instalação das Dependências

Clone o repositório e instale as bibliotecas necessárias:

git clone https://github.com/seu-usuario/tokenwise.git
cd tokenwise

pip install pandas openai anthropic google-genai ecologits python-dotenv


3. Configuração de Variáveis de Ambiente

Crie um arquivo .env na raiz do projeto (baseado no .env.example) e adicione suas chaves de API:

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIzaSy...


⚠️ Atenção: Nunca faça commit do seu arquivo .env no Git.

📊 Coleta e Geração de Dados

Para gerar a base de treinamento do modelo de ML:

Preencha o arquivo prompts_entrada_exemplo.csv com as duplas de prompts originais e otimizados.

Execute o script principal:

python gerar_dataset_tokenwise.py


O script testará automaticamente as duas versões do prompt na API configurada, capturando:

Tokens de entrada e saída

Latência (segundos)

Consumo de energia (kWh) (via EcoLogits)

Consumo de água estimado (Litros) (via EcoLogits)

Emissão de carbono (kg CO₂e) (via EcoLogits)

Os dados consolidados serão salvos incrementalmente no arquivo dataset_tokenwise_gerado.csv.

📈 Métricas do Projeto

Redução de Tokens de Entrada (%): Economia direta no tamanho do contexto enviado.

Taxa de Compressão Segura: Proporção de casos onde a resposta enxuta manteve a qualidade aceitável.

Pegada Ambiental Estimada: Faixas de economia energética e hídrica decorrentes da redução da carga computacional.

👥 Integrantes do Projeto

Projeto desenvolvido como parte da disciplina/desafio de Inteligência Artificial e Machine Learning.
