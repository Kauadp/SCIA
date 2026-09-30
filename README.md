# 🕵️ SCIA

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)
![Evolution API](https://img.shields.io/badge/Evolution%20API-25D366?logo=whatsapp&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-000000)

Sistema de inteligência artificial desenvolvido para analisar mensagens de um grupo do WhatsApp, identificar padrões de comunicação e gerar intervenções automatizadas de acordo com o comportamento observado.

O projeto combina **NLP, embeddings, TF-IDF, aprendizado supervisionado e geração de texto por LLM** para observar mensagens em tempo real e decidir quando uma intervenção do SCIA deve acontecer.

---

# 📖 Sobre o projeto

O **SCIA** nasceu a partir da análise histórica de mensagens de um grupo do WhatsApp, buscando responder a uma pergunta simples:

> **É possível identificar quem escreveu uma mensagem apenas observando a forma como essa pessoa escreve?**

Para investigar essa hipótese, foi construída uma pipeline de Machine Learning utilizando aproximadamente **100 mil mensagens** de **13 participantes**.

O projeto evoluiu de um experimento offline para um sistema capaz de receber mensagens diretamente do WhatsApp, realizar previsões e, quando uma ocorrência atende às regras de negócio, gerar uma resposta utilizando uma LLM.

A evolução do projeto ocorreu em duas versões principais:

- **V1:** classificação de mensagens individuais;
- **V2:** classificação de grupos de cinco mensagens, buscando reduzir o ruído de mensagens isoladas e obter uma assinatura estilística mais representativa do autor.

---

# 🎯 Objetivos

- Investigar se padrões linguísticos permitem identificar os autores das mensagens;
- construir um classificador multiclasse para prever o autor;
- analisar os padrões de comunicação presentes no grupo;
- identificar situações em que o modelo possui maior ou menor confiança;
- estudar a relação entre confiança e acerto;
- criar um sistema capaz de processar mensagens em tempo real;
- gerar intervenções automatizadas utilizando uma LLM;
- observar o comportamento do modelo em produção antes de habilitar o envio automático.

---

# 🧠 Evolução da Pipeline

## V1: mensagens individuais

A primeira versão tratava cada mensagem individualmente como uma observação.

A pipeline combinava:

- embeddings;
- TF-IDF por caracteres;
- características temporais;
- características comportamentais;
- clustering;
- MLP.

O objetivo era identificar o autor mais provável de cada mensagem.

Essa abordagem apresentou desempenho limitado, especialmente devido ao ruído natural de mensagens curtas e ambíguas.

## V2: grupos de cinco mensagens

A segunda versão modificou a unidade de classificação.

Em vez de classificar uma mensagem isolada, as mensagens de cada participante são agrupadas em blocos cronológicos de cinco mensagens.

Cada grupo é representado por:

- concatenação das cinco mensagens;
- início e fim do intervalo temporal;
- duração da sequência;
- horário inicial e final;
- dia da semana inicial e final;
- quantidade de palavras;
- quantidade de caracteres;
- caracteres por palavra.

Essa mudança busca fornecer ao modelo uma quantidade maior de evidência estilística por previsão.

### Configuração selecionada da V2

A análise de ablação indicou que o clustering não contribuía positivamente para o desempenho final, enquanto os embeddings continuavam fornecendo informação relevante.

A configuração selecionada foi:

```text
TF-IDF por caracteres
        +
Embeddings
        +
Características temporais
        +
Características comportamentais
        ↓
       MLP
```

O clustering permanece disponível para análise exploratória e interpretação, mas **não faz parte da pipeline de inferência da V2 selecionada**.

---

# 🧠 Pipeline de Machine Learning

## 1. Processamento das mensagens

As mensagens históricas são processadas para remover elementos que não contribuem para a identificação do autor.

Entre os tratamentos realizados estão:

- remoção de URLs;
- tratamento de mensagens multimídia;
- normalização do texto;
- criação de variáveis temporais;
- características relacionadas ao tamanho das mensagens.

Na V2, essas informações são agregadas ao nível do grupo de cinco mensagens.

São utilizadas características como:

- dia da semana;
- horário;
- duração do grupo;
- quantidade de caracteres;
- quantidade de palavras;
- caracteres por palavra.

A duração apresenta forte assimetria e é transformada para `log_duracao`, reduzindo o efeito de grupos com intervalos excepcionalmente longos.

## 2. Representação textual

### Embeddings

Foi utilizado:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Cada unidade textual é representada por um vetor de **384 dimensões**.

### TF-IDF

A principal representação textual utiliza **n-gramas de caracteres**:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=3,
    max_features=100_000,
    sublinear_tf=True
)
```

A representação por caracteres permite capturar abreviações, erros de digitação, gírias, pontuação e padrões recorrentes de escrita.

---

# 🔬 Clustering

O clustering foi utilizado durante a investigação das características do conjunto de dados e como feature em versões intermediárias do classificador.

O processo utiliza:

- **UMAP** para redução de dimensionalidade;
- **K-Means** para agrupamento.

Na V1, a configuração final utilizada no classificador possuía **25 clusters**.

Na V2, diferentes valores de K foram avaliados e **15 clusters** apresentaram melhor separação segundo a análise exploratória. Entretanto, a ablação mostrou que remover a feature de cluster melhorava o desempenho do classificador.

Por isso, o clustering foi removido da configuração final de inferência da V2.

Ele continua sendo utilizado para:

- exploração dos dados;
- interpretação dos padrões linguísticos;
- análise de grupos semânticos;
- investigação do comportamento do conjunto de mensagens.

### ⚠️ Artefato pesado do UMAP

O artefato final do UMAP possui aproximadamente **458 MB** e não é versionado no GitHub.

Durante o deploy da aplicação, o arquivo é transferido diretamente da máquina de desenvolvimento para a VM utilizando `scp` e colocado no diretório `artifacts/clustering/` quando necessário.

Essa decisão evita ultrapassar o limite de tamanho de arquivos do GitHub e mantém os demais artefatos versionados normalmente.

O arquivo está listado no `.gitignore`.

---

# 🤖 Modelo de classificação

Foram comparados diferentes modelos, incluindo:

- Logistic Regression;
- Random Forest;
- XGBoost;
- MLP.

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

Treinamento:

- AdamW;
- learning rate `0.001`;
- weight decay `1e-4`;
- batch size `256`;
- pesos de classe balanceados;
- early stopping;
- random state `42`.

Na V2, a matriz final possui **100.404 features**:

```text
17 features temporais
+
3 features comportamentais
+
384 embeddings
+
100.000 features TF-IDF
=
100.404 features
```

---

# 📊 Resultados Offline

## V1 — Mensagens individuais

O modelo final da V1 apresentou no conjunto de teste histórico:

| Métrica | V1 |
|---|---:|
| Accuracy | **34,0%** |
| Macro F1 | **30,2%** |
| Weighted F1 | **35,0%** |

O problema possui **13 classes**, com diferentes níveis de separabilidade entre os participantes.

A análise posterior mostrou que a confiança do modelo continha informação útil sobre a correção das previsões, mas grande parte das mensagens apresentava baixa confiança.

A V1 serviu como baseline para a evolução do projeto.

## V2 — Grupos de cinco mensagens

A V2 apresentou uma melhoria substancial em relação à abordagem de mensagens individuais.

A configuração selecionada, sem clustering e com embeddings, apresentou:

| Métrica | V2 |
|---|---:|
| Accuracy | **64,0%** |
| Macro F1 | **56,9%** |
| Weighted F1 | **64,0%** |

Melhoria de Macro F1 em relação à V1:

```text
V1: 30,2%
V2: 56,9%

+26,7 pontos percentuais
```

A melhoria está associada principalmente à mudança da unidade de classificação: a V2 utiliza grupos de cinco mensagens, fornecendo ao modelo uma quantidade maior de evidência estilística por observação.

### Ablação da V2

Foram avaliadas diferentes combinações de features:

| Configuração | Macro F1 | Accuracy |
|---|---:|---:|
| V2 com clustering + embeddings | 55,3% | 62% |
| **V2 sem clustering + embeddings** | **56,9%** | **64%** |
| V2 sem clustering + sem embeddings | 53,9% | 59% |

A análise indica que:

- o clustering não melhorou o desempenho final;
- os embeddings contribuíram positivamente;
- a configuração selecionada combina TF-IDF, embeddings, características temporais e características comportamentais.

---

# 🎯 Confiança e Seleção de Previsões na V2

A análise de confiança mostrou uma relação clara entre a probabilidade máxima do modelo e a taxa de acerto.

| Faixa de confiança | Acurácia | N |
|---|---:|---:|
| ≤ 0,30 | 25,5% | 102 |
| 0,30–0,40 | 28,2% | 280 |
| 0,40–0,50 | 40,2% | 393 |
| 0,50–0,60 | 42,8% | 542 |
| 0,60–0,70 | 55,3% | 461 |
| 0,70–0,80 | 67,4% | 488 |
| 0,80–0,90 | **75,3%** | **575** |
| 0,90–1,00 | **92,8%** | **1.108** |

A análise de Precisão × Cobertura mostrou aproximadamente:

| Cobertura | Precisão |
|---:|---:|
| ~19% | ~95,2% |
| ~42,6% | ~86,8% |
| ~61,4% | ~79,9% |
| ~80,4% | ~71,8% |
| ~97,4% | ~65,3% |

Esse comportamento motivou a utilização de **P ≥ 0,80 como threshold da regra de negócio**.

O objetivo não é eliminar todos os erros do classificador, mas retirar da camada de intervenção os casos nos quais a evidência estilística é insuficiente.

A partir do threshold, os casos são classificados em:

```text
0,80 ≤ P < 0,90
    ├── acerto → ACERTO_BAIXO
    └── erro   → ERRO_BAIXO

0,90 ≤ P ≤ 1,00
    ├── acerto → ACERTO_ALTO
    └── erro   → ERRO_ALTO
```

Os eventos de acerto são registrados para análise, mas **não acionam a LLM**.

Apenas `ERRO_BAIXO` e `ERRO_ALTO` são candidatos a intervenção, respeitando ainda o cooldown individual de **5 minutos por pessoa**.

---

# 🧩 Arquitetura

O sistema é dividido em três estágios principais:

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
    ├── Embeddings
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

O terceiro estágio permanece separado da geração das mensagens para permitir observar o sistema em produção sem enviar automaticamente as respostas.

---

# 🔄 Fluxo de processamento

## 1. Recebimento

A **Evolution API** recebe as mensagens do WhatsApp e envia os eventos para o webhook desenvolvido em FastAPI.

O webhook:

- valida o grupo;
- identifica o participante;
- extrai a mensagem;
- converte o timestamp;
- armazena o conteúdo no banco.

## 2. Predição

Na V1, cada mensagem era tratada individualmente.

Na V2, as mensagens são consolidadas em grupos cronológicos de cinco mensagens por participante antes da classificação.

Para cada grupo são executadas:

1. extração das características;
2. transformação do texto;
3. geração do embedding;
4. transformação TF-IDF;
5. cálculo das features temporais e comportamentais;
6. previsão do autor;
7. cálculo da probabilidade;
8. classificação da confiança.

O resultado é armazenado em `previsoes`.

## 3. Geração da resposta

O segundo worker busca previsões elegíveis.

A LLM recebe apenas o contexto operacional definido para o evento de erro, incluindo:

- categoria;
- personalidade selecionada;
- instruções de geração.

Ela **não recebe o autor real, a probabilidade da previsão ou as features utilizadas pelo classificador**.

A decisão sobre a existência do erro pertence ao classificador e às regras de negócio.

## 4. Envio

O terceiro worker busca mensagens geradas e ainda não enviadas.

A mensagem é enviada através da Evolution API e seu status é atualizado no banco.

Durante a fase atual de observação, esse worker permanece desativado.

---

# 🤖 Geração de mensagens

A geração utiliza a API da **Groq**, com o modelo:

```text
openai/gpt-oss-20b
```

O sistema possui sete personalidades:

- ansioso;
- dramático;
- caótico;
- gaúcho;
- mineiro;
- conspiracionista;
- fofo.

As personalidades são combinadas com as categorias produzidas pelo classificador, permitindo diferentes estilos de resposta.

A LLM é responsável **exclusivamente pela geração do texto**. A decisão de intervir pertence ao pipeline de classificação e às regras de negócio.

---

# 🗄️ Banco de dados

O PostgreSQL armazena o fluxo de processamento em três tabelas principais.

## `mensagens_raw`

```text
id
membro
data_hora
mensagem
processada
```

Armazena as mensagens recebidas do WhatsApp.

## `previsoes`

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

Armazena o resultado do modelo de Machine Learning.

## `mensagens_bot`

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

Armazena as mensagens geradas pela LLM e seu status de envio.

---

# 🐳 Docker e infraestrutura

A aplicação é executada utilizando Docker Compose.

A infraestrutura possui:

- Evolution API;
- Redis;
- PostgreSQL;
- webhook FastAPI.

Todos os serviços pertencem à mesma rede Docker e se comunicam internamente.

```text
Evolution API
      │
      ▼
webhook:8000
      │
      ▼
PostgreSQL
```

## Python 3.12

A imagem do webhook utiliza Python 3.12.

A mudança de Python 3.11 para 3.12 foi necessária porque o artefato `umap_cluster.joblib` apresentou incompatibilidade ao ser carregado no ambiente Python 3.11, apesar das versões das principais bibliotecas serem equivalentes.

O artefato havia sido gerado no ambiente Python 3.12.

Com Python 3.12, o carregamento do UMAP foi normalizado.

## Desempenho observado do container

Na execução local, o consumo de memória observado foi aproximadamente:

| Serviço | Memória |
|---|---:|
| `scia_webhook` | 1,87 GiB |
| `scia_evolution` | 243 MiB |
| `scia_evolution_postgres` | 38 MiB |
| `scia_evolution_redis` | 12 MiB |
| **Total aproximado** | **2,16 GiB** |

O container do webhook concentra a maior parte do consumo devido ao carregamento do pipeline de NLP/ML, especialmente PyTorch, Sentence Transformers e os artefatos do modelo.

A primeira construção completa da imagem levou aproximadamente **20 minutos** e a imagem Docker chegou a ocupar cerca de **11,6 GB** de espaço local.

Esses valores são referências do ambiente de desenvolvimento e podem variar conforme cache, versões das dependências e carga.

## Oracle Cloud

A arquitetura foi construída para execução em uma VM Oracle Cloud.

A experiência inicial mostrou que uma máquina com apenas **1 GB de RAM não é confortável para a versão atual** do SCIA, principalmente devido ao carregamento dos artefatos de NLP/ML.

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
│   ├── analysis/
│   ├── clustering/
│   ├── embedding/
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
│   ├── 05_model_analysis.ipynb
│   └── 06_prod_model_analysis.ipynb
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 📂 Notebooks

| Notebook | Descrição |
|---|---|
| `01_eda_processamento.ipynb` | Análise exploratória e processamento dos dados |
| `02_embeddings_tfidf.ipynb` | Geração de embeddings e representações TF-IDF |
| `03_clustering.ipynb` | UMAP e K-Means |
| `04_modeling.ipynb` | Treinamento e comparação dos modelos |
| `05_model_analysis.ipynb` | Análise detalhada dos resultados offline e erros |
| `06_prod_model_analysis.ipynb` | Monitoramento contínuo do comportamento do modelo em produção |

## `06_prod_model_analysis.ipynb`

O sexto notebook foi criado para transformar a análise de produção em um processo reexecutável. Ele lê diretamente as tabelas do banco e permite acompanhar a evolução do SCIA conforme novas mensagens são acumuladas.

O notebook está dividido em cinco frentes:

### 1. Saúde do Pipeline e SLAs

- conversão entre `mensagens_raw`, `previsoes` e `mensagens_bot`;
- retenção do pré-processamento;
- intervalos de confiança para proporções;
- latência do Worker ML;
- latência da LLM;
- preparação para medir a latência do Worker 3;
- distribuição dos status.

### 2. Desempenho e confiança do modelo

- Accuracy;
- Macro F1;
- Weighted F1;
- classification report por membro;
- matriz de confusão;
- distribuição das probabilidades;
- teste de Mann-Whitney entre probabilidades de acertos e erros;
- tamanho de efeito rank-biserial.

O teste de Mann-Whitney é utilizado para verificar se previsões corretas tendem a apresentar probabilidades maiores que previsões incorretas. Essa análise avalia a relação entre **confiança e acerto**, e não constitui uma avaliação formal de calibração probabilística.

### 3. Impostores e Doppelgängers

- taxa geral de acertos e erros;
- frequência de previsões incorretas por autor real;
- identidades mais frequentemente atribuídas pelo modelo;
- distribuição das categorias de intervenção;
- matriz de Doppelgängers;
- identificação dos principais pares de confusão.

### 4. Monitoramento da LLM e Delivery

- quantidade de textos gerados;
- tamanho das respostas;
- frequência das personalidades;
- variabilidade de tamanho por personalidade;
- integridade dos textos;
- status de envio;
- latência da Groq;
- preparação para monitorar o Worker 3 após sua ativação.

### 5. Drift e mudança temporal

- volume de mensagens por horário;
- distribuição de características das mensagens;
- Macro F1 por semana;
- comparação entre janelas temporais;
- baseline da primeira semana;
- testes de mudança de distribuição quando houver múltiplas semanas.

---

# 📈 Resultados em Produção

Os resultados de produção estão sendo acompanhados separadamente dos resultados offline.

A V1 possui uma janela de produção em coleta para permitir uma análise mais representativa de meia semana completa.

## V1 — Produção

| Métrica | V1 — Produção |
|---|---:|
| Período analisado | **4 Dias** |
| Previsões | **1.849** |
| Accuracy | **27,91%** |
| Macro F1 | **0,2446** |
| Weighted F1 | **0,2927** |
| Mediana de confiança — acertos | **0,6550** |
| Mediana de confiança — erros | **0,4206** |
| Intervenções geradas | **124** |
| Mensagens acima do threshold (P≥0,70) | **24,12%** |
| Taxa de intervenção | **6,71%** |
| Latência ML | **mediana 6,14s** |
| Latência LLM | **mediana 1,81s** |
| Delivery | **mediana 33,86s** |

A análise será consolidada após uma semana completa de observação, permitindo comparar o comportamento real da V1 com o teste histórico.

## V2 — Produção

> **Status: ainda não avaliada em produção.**

A V2 foi validada offline utilizando grupos de cinco mensagens. Antes de sua entrada em produção, será necessário garantir que a estratégia de inferência reproduza a mesma unidade de classificação utilizada no treinamento.

| Métrica | V2 — Produção |
|---|---:|
| Período analisado | **A preencher** |
| Grupos avaliados | **A preencher** |
| Accuracy | **A preencher** |
| Macro F1 | **A preencher** |
| Weighted F1 | **A preencher** |
| Mediana de confiança — acertos | **A preencher** |
| Mediana de confiança — erros | **A preencher** |
| Cobertura `P ≥ 0,80` | **A preencher** |
| Intervenções geradas | **A preencher** |
| Taxa de intervenção | **A preencher** |
| Latência ML | **A preencher** |
| Latência LLM | **A preencher** |
| Delivery | **A preencher após ativação do Worker 3** |

---

# 🧪 Modo de observação

Durante a validação inicial:

```text
Worker 1 → ativo

Worker 2 → ativo

Worker 3 → desativado
```

Isso permite acompanhar:

- distribuição das previsões;
- nível de confiança;
- frequência de intervenções;
- distribuição entre ACERTO e ERRO;
- principais confusões;
- comportamento da LLM;
- latências do processamento.

O Worker 3 será habilitado somente depois que houver evidência suficiente para avaliar o comportamento do sistema em produção.

---

# 💡 Ideia central

O SCIA combina três problemas diferentes:

```text
┌───────────────────────┐
│   Machine Learning    │
│                       │
│ Quem escreveu isso?   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│       Confiança       │
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

Na V2, essa separação é complementada por uma regra de negócio que utiliza `P ≥ 0,80` como crivo mínimo de confiança e um cooldown de cinco minutos por pessoa para controlar a frequência das intervenções.

---

# 🚀 Próximos passos

- finalizar a coleta da primeira semana completa de produção da V1;
- atualizar as métricas de produção da V1;
- comparar V1 offline × V1 produção;
- acumular dados para avaliar estabilidade temporal;
- acompanhar a relação entre confiança e acerto;
- acompanhar os principais Doppelgängers;
- validar a nova pipeline V2 em produção;
- comparar V2 offline × V2 produção;
- avaliar possíveis ajustes no threshold somente após maior volume de dados;
- habilitar o Worker 3 após a fase de observação;
- monitorar latência e sucesso real de delivery;
- avaliar o deploy definitivo em VM após validar os requisitos computacionais.

---

# 👨‍💻 Autor

**Kauã Dias**

Estudante de Estatística — Universidade Federal do Espírito Santo (UFES)

GitHub: https://github.com/kauadp

LinkedIn: https://linkedin.com/in/kauad

---

# 📄 Licença

Projeto desenvolvido para fins de estudo e experimentação em **Machine Learning, NLP, classificação de autoria, sistemas de intervenção automatizada e integração com LLMs**.

Os dados originais utilizados no treinamento não são disponibilizados publicamente por conterem mensagens privadas de um grupo do WhatsApp.
