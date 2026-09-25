# 🕵️ SCIA

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi\&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch\&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker\&logoColor=white)
![Evolution API](https://img.shields.io/badge/Evolution%20API-25D366?logo=whatsapp\&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-000000)

Sistema de inteligência artificial desenvolvido para analisar mensagens de um grupo do WhatsApp, identificar padrões de comunicação e gerar intervenções automatizadas de acordo com o comportamento observado.

O projeto combina **NLP, embeddings, TF-IDF, clustering, aprendizado supervisionado e geração de texto por LLM** para construir um sistema capaz de observar as mensagens do grupo em tempo real e decidir quando uma intervenção do SCIA deve acontecer.

---

# 📖 Sobre o projeto

O **SCIA** nasceu a partir da análise histórica de mensagens de um grupo do WhatsApp, buscando responder a uma pergunta simples:

> **É possível identificar quem escreveu uma mensagem apenas observando a forma como essa pessoa escreve?**

Para investigar essa hipótese, foi construída uma pipeline de Machine Learning utilizando aproximadamente **100 mil mensagens** de 13 participantes.

O projeto passou por diferentes etapas de processamento e modelagem, incluindo:

* análise exploratória dos dados;
* limpeza e processamento das mensagens;
* geração de embeddings;
* TF-IDF;
* redução de dimensionalidade;
* clustering;
* criação de variáveis temporais;
* treinamento de modelos de classificação;
* análise de erros e confiança das previsões;
* integração do modelo a um sistema em tempo real.

Após a etapa de modelagem, o projeto evoluiu de um experimento offline para um sistema capaz de receber mensagens diretamente do WhatsApp, realizar a previsão e, quando a confiança é suficiente, gerar uma resposta utilizando uma LLM.

---

# 🎯 Objetivos

* Investigar se padrões linguísticos permitem identificar os autores das mensagens;
* Construir um modelo de classificação multiclasse para prever o autor de uma mensagem;
* Analisar os padrões de comunicação presentes no grupo;
* Identificar situações em que o modelo possui maior ou menor confiança;
* Criar um sistema capaz de processar mensagens em tempo real;
* Gerar intervenções automatizadas utilizando uma LLM;
* Avaliar o comportamento do modelo em produção antes de habilitar o envio automático das mensagens.

---

# 🧠 Pipeline de Machine Learning

O processo de modelagem foi dividido em diferentes etapas.

## 1. Processamento das mensagens

As mensagens históricas são inicialmente processadas para remover elementos que não contribuem para a identificação do autor.

Entre os tratamentos realizados estão:

* remoção de URLs;
* tratamento de mensagens multimídia;
* normalização do texto;
* criação de variáveis temporais;
* cálculo de características relacionadas ao tamanho das mensagens.

Também são utilizadas informações como:

* dia da semana;
* horário;
* quantidade de caracteres;
* quantidade de palavras;
* caracteres por palavra.

---

## 2. Representação textual

Foram avaliadas diferentes formas de representar as mensagens.

### Embeddings

Foi utilizado o modelo:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Cada mensagem é representada por um vetor de **384 dimensões**.

### TF-IDF

Também foram testadas representações TF-IDF.

A representação que apresentou melhor desempenho foi baseada em **n-gramas de caracteres**, utilizando:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=3,
    max_features=100_000,
    sublinear_tf=True
)
```

A utilização de caracteres permite capturar características como:

* abreviações;
* erros de digitação;
* padrões de escrita;
* gírias;
* pontuação;
* formas recorrentes de escrever determinadas palavras.

---

# 🔬 Clustering

Além da classificação supervisionada, o projeto utiliza clustering para identificar padrões de comportamento nas mensagens.

O processo utiliza:

* UMAP para redução de dimensionalidade;
* K-Means para agrupamento;
* 25 clusters na versão final do modelo.

Os clusters são utilizados como uma das características fornecidas ao classificador.

---

# 🤖 Modelo de classificação

Foram comparados diferentes modelos de Machine Learning, incluindo:

* Logistic Regression;
* Random Forest;
* XGBoost;
* MLP.

O modelo final utiliza uma **Multilayer Perceptron (MLP)** implementada em PyTorch.

Arquitetura:

```text
Entrada
   │
   ▼
256 neurônios
   │
 ReLU
   │
Dropout 0.4
   │
   ▼
128 neurônios
   │
 ReLU
   │
Dropout 0.3
   │
   ▼
13 classes
```

O treinamento utiliza:

* AdamW;
* learning rate de `0.001`;
* weight decay de `1e-4`;
* batch size de `256`;
* pesos de classe balanceados;
* early stopping.

---

# 📊 Resultados

O modelo final apresentou aproximadamente:

| Métrica     | Resultado |
| ----------- | --------- |
| Accuracy    | 34%       |
| Macro F1    | 30,2%     |
| Weighted F1 | 35%       |

O problema possui **13 classes**, portanto a classificação apresenta um nível considerável de dificuldade.

A análise dos erros mostrou que algumas pessoas possuem padrões de escrita mais facilmente distinguíveis, enquanto outras apresentam características muito semelhantes.

Entre as principais confusões observadas estão:

* Caio ↔ Luquete;
* Luquete ↔ Luiz;
* Arthur → Guido / Ítalo / Cauã;
* Jão → Guido / Cauã.

O objetivo do modelo não é simplesmente produzir uma previsão para todas as mensagens, mas também estimar **o nível de confiança dessa previsão**.

---

# 🎯 Sistema de confiança

A probabilidade produzida pelo classificador é utilizada para determinar se uma mensagem deve ou não gerar uma intervenção.

| Probabilidade     | Classificação |
| ----------------- | ------------- |
| `< 0.70`          | Ignorar       |
| `0.70 ≤ p < 0.80` | BAIXO         |
| `0.80 ≤ p < 0.90` | MEDIO         |
| `0.90 ≤ p ≤ 1.00` | ALTO          |

Quando a confiança é inferior a `0.70`, nenhuma intervenção é gerada.

Quando a mensagem ultrapassa esse limite, ela recebe uma das seguintes classificações:

```text
ACERTO_BAIXO
ACERTO_MEDIO
ACERTO_ALTO

ERRO_BAIXO
ERRO_MEDIO
ERRO_ALTO
```

As classificações de **ACERTO** e **ERRO** são possíveis porque o sistema possui acesso ao autor real da mensagem durante a fase de análise.

---

# 🧩 Arquitetura

O sistema é dividido em três etapas principais:

```text
WhatsApp
    │
    ▼
Evolution API
    │
    ▼
Webhook FastAPI
    │
    ▼
mensagens_raw
    │
    ▼
Worker 1
    │
    ├── Pré-processamento
    ├── Features
    ├── TF-IDF
    ├── Clustering
    └── Classificação
    │
    ▼
previsoes
    │
    ▼
Worker 2
    │
    └── Geração com LLM
    │
    ▼
mensagens_bot
    │
    ▼
Worker 3
    │
    └── Evolution API
    │
    ▼
WhatsApp
```

O terceiro estágio permanece separado da geração das mensagens para permitir que o sistema seja observado em produção sem enviar automaticamente as respostas.

---

# 🔄 Fluxo de processamento

## 1. Recebimento

A **Evolution API** recebe as mensagens do WhatsApp e envia os eventos para o webhook desenvolvido em FastAPI.

O webhook:

* valida o grupo;
* identifica o participante;
* extrai a mensagem;
* converte o timestamp;
* armazena o conteúdo no banco.

---

## 2. Predição

O primeiro worker busca mensagens ainda não processadas.

Para cada mensagem são executadas:

1. extração das características;
2. transformação do texto;
3. identificação do cluster;
4. previsão do autor;
5. cálculo da probabilidade;
6. classificação da confiança.

O resultado é armazenado na tabela `previsoes`.

---

## 3. Geração da resposta

O segundo worker busca previsões elegíveis.

A categoria da previsão é utilizada para determinar o comportamento da LLM.

A LLM recebe:

* categoria;
* personalidade selecionada;
* instruções de geração.

Ela **não recebe o autor real**, nem a probabilidade da previsão ou as características utilizadas pelo modelo.

Isso mantém a geração da resposta separada da lógica de classificação.

---

## 4. Envio

O terceiro worker busca mensagens geradas e ainda não enviadas.

A mensagem é enviada através da Evolution API e seu status é atualizado no banco.

Esse worker pode permanecer desativado durante os testes, permitindo observar o comportamento do sistema sem gerar mensagens no grupo.

---

# 🤖 Geração de mensagens

A geração de texto utiliza a API da **Groq**, com o modelo:

```text
openai/gpt-oss-20b
```

O sistema possui diferentes personalidades para controlar o estilo das respostas.

Entre elas:

* ansioso;
* dramático;
* caótico;
* gaúcho;
* mineiro;
* conspiracionista;
* fofo.

As personalidades são combinadas com as categorias produzidas pelo classificador, permitindo diferentes comportamentos para cada situação.

Exemplo:

```text
ERRO_ALTO + conspiracionista
ERRO_MEDIO + caótico
ACERTO_ALTO + dramático
```

O resultado é uma resposta gerada dinamicamente pela LLM.

---

# 🗄️ Banco de dados

O PostgreSQL é utilizado para armazenar todo o fluxo de processamento.

## `mensagens_raw`

Armazena as mensagens recebidas do WhatsApp.

```text
id
membro
data_hora
mensagem
processada
```

---

## `previsoes`

Armazena o resultado do modelo de Machine Learning.

```text
id
mensagem_id
mensagem
autor_real
autor_predito
probabilidade
categoria
features
status_bot
processado_em
```

---

## `mensagens_bot`

Armazena as mensagens geradas pela LLM e seu status de envio.

```text
id
previsao_id
categoria
personalidade
texto
status
gerado_em
enviado_em
```

---

# 🐳 Docker

A aplicação é executada utilizando Docker.

A infraestrutura da Evolution API possui serviços para:

* Evolution API;
* Redis;
* PostgreSQL.

O webhook FastAPI também é executado dentro da mesma rede Docker.

Dessa forma, a comunicação entre os serviços ocorre internamente:

```text
Evolution API
      │
      ▼
webhook:8000
      │
      ▼
PostgreSQL
```

Isso permite que o projeto seja executado em uma máquina local ou posteriormente migrado para uma VM sem alterar a arquitetura da aplicação.

---

# 🏗 Estrutura do projeto

```text
SCIA/

├── app/
│   ├── banco/
│   │   ├── conexao.py
│   │   └── tabelas.sql
│   │
│   ├── evolution/
│   │   ├── cliente.py
│   │   └── docker-compose.yml
│   │
│   ├── webhook/
│   │   ├── main.py
│   │   ├── maps.py
│   │   └── parser.py
│   │
│   ├── modelo/
│   │   ├── features.py
│   │   ├── modelo.py
│   │   ├── preditor.py
│   │   └── preprocessamento.py
│   │
│   ├── llm/
│   │   ├── cliente.py
│   │   ├── gerador.py
│   │   ├── prompts.py
│   │   └── __init__.py
│   │
│   ├── controlador.py
│   └── worker.py
│
├── artifacts/
│   ├── clustering/
│   ├── models/
│   └── tfidf/
│
├── dados/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda_processamento.ipynb
│   ├── 02_embeddings_tfidf.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_model_analysis.ipynb
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 📂 Notebooks

O desenvolvimento do modelo foi organizado em notebooks separados por etapa.

| Notebook                     | Descrição                                      |
| ---------------------------- | ---------------------------------------------- |
| `01_eda_processamento.ipynb` | Análise exploratória e processamento dos dados |
| `02_embeddings_tfidf.ipynb`  | Geração de embeddings e representações TF-IDF  |
| `03_clustering.ipynb`        | UMAP e K-Means                                 |
| `04_modeling.ipynb`          | Treinamento e comparação dos modelos           |
| `05_model_analysis.ipynb`    | Análise detalhada dos resultados e erros       |

---

# 🧰 Tecnologias

| Tecnologia            | Utilização                      |
| --------------------- | ------------------------------- |
| Python                | Desenvolvimento geral           |
| FastAPI               | Webhook e API                   |
| PostgreSQL            | Banco de dados                  |
| PyTorch               | Modelo MLP                      |
| scikit-learn          | TF-IDF, modelos e métricas      |
| XGBoost               | Modelo de comparação            |
| UMAP                  | Redução de dimensionalidade     |
| K-Means               | Clustering                      |
| Sentence Transformers | Embeddings                      |
| Groq                  | Geração de texto                |
| Evolution API         | Integração com WhatsApp         |
| Docker                | Containerização                 |
| Redis                 | Infraestrutura da Evolution API |

---

# 🔐 Variáveis de ambiente

As credenciais e configurações sensíveis são mantidas em arquivos `.env` e não devem ser versionadas.

Principais configurações utilizadas pelo projeto:

```text
EVOLUTION_API_URL
EVOLUTION_API_KEY
EVOLUTION_INSTANCE

db_uri

GROQ_API_KEY
```

Os arquivos `.env` estão incluídos no `.gitignore`.

---

# 📈 Resultados em produção

Após a integração com o WhatsApp, o sistema passou a processar mensagens reais do grupo.

Em uma primeira amostra de **47 mensagens recebidas**, 46 chegaram à etapa de previsão e 13 foram consideradas elegíveis para geração de resposta utilizando o limite de confiança de `0.70`.

Isso correspondeu a aproximadamente:

```text
46 previsões
13 intervenções elegíveis
≈ 28% das mensagens
```

A distribuição observada na primeira amostra foi:

```text
ERRO_MEDIO   5
ERRO_BAIXO   3
ACERTO_ALTO  3
ERRO_ALTO    1
ACERTO_BAIXO 1
```

Os resultados iniciais ficaram próximos do comportamento observado durante os testes offline do modelo.

Por se tratar de uma amostra pequena, a avaliação definitiva do comportamento em produção depende da coleta de um volume maior de mensagens.

---

# 🧪 Modo de observação

Durante a validação inicial, o sistema pode operar sem enviar mensagens ao grupo.

Nesse modo:

```text
Worker 1 → ativo
Worker 2 → ativo
Worker 3 → desativado
```

Isso permite acompanhar:

* distribuição das previsões;
* nível de confiança;
* frequência de intervenções;
* distribuição entre ACERTO e ERRO;
* principais confusões;
* comportamento da geração da LLM.

Após a validação, o terceiro worker pode ser habilitado para realizar os envios automaticamente.

---

# 💡 Ideia central

O SCIA não foi desenvolvido apenas como um bot de respostas automáticas.

O projeto combina três problemas diferentes:

```text
┌───────────────────────┐
│      Machine Learning │
│                       │
│ Quem escreveu isso?   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      Sistema de       │
│       confiança       │
│                       │
│ Devo reagir?          │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│          LLM          │
│                       │
│ O que devo responder? │
└───────────────────────┘
```

A separação entre essas etapas permite que o modelo estatístico seja responsável pela **identificação e confiança**, enquanto a LLM fica responsável exclusivamente pela **geração do texto**.

---

# 🚀 Próximos passos

* aumentar o volume de observações em produção;
* comparar a distribuição real e a distribuição do conjunto de teste;
* analisar novamente as principais confusões;
* avaliar possíveis ajustes nos limites de confiança;
* melhorar o modelo de classificação;
* avaliar novas representações textuais;
* acompanhar a qualidade das intervenções geradas;
* habilitar o envio automático após a validação.

---

# 👨‍💻 Autor

**Kauã Dias**

Estudante de Estatística — Universidade Federal do Espírito Santo (UFES)

GitHub:
https://github.com/kauadp

LinkedIn:
https://linkedin.com/in/kauad

---

## 📄 Licença

Projeto desenvolvido para fins de estudo e experimentação em **Machine Learning, NLP, sistemas de recomendação de intervenção e integração com LLMs**.

Os dados originais utilizados no treinamento não são disponibilizados publicamente por conterem mensagens privadas de um grupo do WhatsApp.
