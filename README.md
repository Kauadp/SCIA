# 🕵️ SCIA

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)

![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi\&logoColor=white)

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql\&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn)

![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)

![Evolution API](https://img.shields.io/badge/Evolution%20API-25D366?logo=whatsapp\&logoColor=white)

![Groq](https://img.shields.io/badge/Groq-000000)

Sistema de inteligência artificial desenvolvido para analisar mensagens de um grupo do WhatsApp, identificar padrões de comunicação e gerar intervenções automatizadas de acordo com o comportamento observado.

O projeto combina **NLP, embeddings, TF-IDF, aprendizado supervisionado e geração de texto por LLM** para observar mensagens, identificar possíveis autores e decidir quando uma intervenção do SCIA deve acontecer.

Além da classificação em tempo real, o sistema possui uma camada de produto chamada **Diário do Inspetor**, responsável por transformar os eventos observados diariamente em um relatório narrativo em PDF.

---

# 📖 Sobre o projeto

O **SCIA** nasceu a partir da análise histórica de mensagens de um grupo do WhatsApp, buscando responder a uma pergunta simples:

> **É possível identificar quem escreveu uma mensagem apenas observando a forma como essa pessoa escreve?**

Para investigar essa hipótese, foi construída uma pipeline de Machine Learning utilizando aproximadamente **100 mil mensagens** de **13 participantes**.

O projeto evoluiu de um experimento offline para um sistema operacional conectado ao WhatsApp, capaz de:

* receber mensagens em tempo real;
* armazenar as mensagens em PostgreSQL;
* consolidar mensagens em grupos de cinco;
* gerar features linguísticas, temporais e comportamentais;
* prever o autor mais provável;
* medir a confiança da previsão;
* classificar a ocorrência de acordo com regras de negócio;
* gerar intervenções utilizando uma LLM;
* enviar as mensagens através da Evolution API;
* acompanhar o comportamento do modelo em produção;
* gerar diariamente um PDF com o **Diário do Inspetor**.

A evolução do projeto ocorreu em duas versões principais:

* **V1:** classificação de mensagens individuais;
* **V2:** classificação de grupos de cinco mensagens, buscando reduzir o ruído de mensagens isoladas e obter uma assinatura estilística mais representativa do autor.

---

# 🎯 Objetivos

* Investigar se padrões linguísticos permitem identificar os autores das mensagens;
* construir um classificador multiclasse para prever o autor;
* analisar os padrões de comunicação presentes no grupo;
* identificar situações em que o modelo possui maior ou menor confiança;
* estudar a relação entre confiança e acerto;
* criar um sistema capaz de processar mensagens em tempo real;
* gerar intervenções automatizadas utilizando uma LLM;
* separar identificação estatística de geração de linguagem;
* acompanhar o comportamento do modelo em produção;
* transformar os eventos do sistema em um produto narrativo através do Diário do Inspetor.

---

# 🧠 Evolução da Pipeline

## V1: mensagens individuais

A primeira versão tratava cada mensagem individualmente como uma observação.

A pipeline combinava:

* embeddings;
* TF-IDF por caracteres;
* características temporais;
* características comportamentais;
* clustering;
* MLP.

O objetivo era identificar o autor mais provável de cada mensagem.

No conjunto de teste histórico, a V1 apresentou:

| Métrica     |        V1 |
| ----------- | --------: |
| Accuracy    | **34,0%** |
| Macro F1    | **30,2%** |
| Weighted F1 | **35,0%** |

A abordagem apresentou desempenho limitado principalmente devido ao ruído natural de mensagens curtas e ambíguas.

A V1 serviu como baseline para a evolução do projeto.

---

## V2: grupos de cinco mensagens

A segunda versão modificou a unidade de classificação.

Em vez de classificar uma mensagem isolada, as mensagens de cada participante são agrupadas em blocos cronológicos de cinco mensagens.

Cada grupo é representado por:

* concatenação das cinco mensagens;
* início e fim do intervalo temporal;
* duração da sequência;
* horário inicial e final;
* dia da semana inicial e final;
* quantidade de palavras;
* quantidade de caracteres;
* caracteres por palavra.

Essa mudança fornece ao modelo uma quantidade maior de evidência estilística por previsão.

No conjunto de teste histórico, a configuração selecionada da V2 apresentou:

| Métrica     |        V2 |
| ----------- | --------: |
| Accuracy    | **64,0%** |
| Macro F1    | **56,9%** |
| Weighted F1 | **64,0%** |

A principal mudança entre V1 e V2 não foi apenas a troca de algoritmo ou hiperparâmetros, mas a mudança da **unidade de observação**.

---

## Configuração selecionada da V2

A análise de ablação indicou que o clustering não contribuía positivamente para o desempenho final, enquanto os embeddings forneciam informação adicional.

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

O clustering permanece disponível para análise exploratória e interpretação, mas **não faz parte da pipeline de inferência da V2**.

---

# 🧠 Pipeline de Machine Learning

## 1. Processamento das mensagens

As mensagens históricas são processadas para remover elementos que não contribuem para a identificação do autor.

Entre os tratamentos realizados estão:

* remoção de URLs;
* tratamento de mensagens multimídia;
* normalização do texto;
* criação de variáveis temporais;
* características relacionadas ao tamanho das mensagens.

Na V2, essas informações são agregadas ao nível do grupo de cinco mensagens.

São utilizadas características como:

* dia da semana;
* horário;
* duração do grupo;
* quantidade de caracteres;
* quantidade de palavras;
* caracteres por palavra.

A duração apresenta forte assimetria e é transformada para `log_duracao`, reduzindo o efeito de grupos com intervalos excepcionalmente longos.

---

## 2. Representação textual

### Embeddings

Foi utilizado:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Cada unidade textual é representada por um vetor de **384 dimensões**.

Os embeddings são utilizados como uma representação semântica complementar à representação baseada em caracteres.

---

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

A representação por caracteres permite capturar:

* abreviações;
* erros de digitação;
* gírias;
* pontuação;
* padrões ortográficos;
* formas recorrentes de escrita.

Na análise de ablação da V2, o bloco de TF-IDF apresentou maior contribuição relativa que embeddings na explicação dos resultados do modelo. Essa análise é descritiva e não deve ser interpretada como uma medida causal da importância de cada feature.

---

# 🔬 Clustering

O clustering foi utilizado durante a investigação das características do conjunto de dados e como feature em versões intermediárias do classificador.

O processo utiliza:

* **UMAP** para redução de dimensionalidade;
* **K-Means** para agrupamento.

Na V1, a configuração final utilizada no classificador possuía **25 clusters**.

Na V2, diferentes valores de K foram avaliados e **15 clusters** apresentaram melhor separação segundo a análise exploratória.

Entretanto, a ablação mostrou que remover a feature de cluster melhorava o desempenho do classificador.

Por isso, o clustering foi removido da configuração final de inferência da V2.

Ele continua sendo utilizado para:

* exploração dos dados;
* interpretação dos padrões linguísticos;
* análise de grupos semânticos;
* investigação do comportamento do conjunto de mensagens.

### ⚠️ Artefato pesado do UMAP

O artefato final do UMAP possui aproximadamente **458 MB** e não é versionado no GitHub.

Quando necessário, o arquivo é transferido diretamente da máquina de desenvolvimento para a VM utilizando `scp` e colocado no diretório:

```text
artifacts/clustering/
```

Essa decisão evita ultrapassar o limite de tamanho de arquivos do GitHub.

---

# 🤖 Modelo de classificação

Foram comparados diferentes modelos, incluindo:

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

Treinamento:

* AdamW;
* learning rate `0.001`;
* weight decay `1e-4`;
* batch size `256`;
* pesos de classe balanceados;
* early stopping;
* random state `42`.

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

| Métrica     |        V1 |
| ----------- | --------: |
| Accuracy    | **34,0%** |
| Macro F1    | **30,2%** |
| Weighted F1 | **35,0%** |

O problema possui **13 classes**, com diferentes níveis de separabilidade entre os participantes.

A análise posterior mostrou que a confiança do modelo continha informação útil sobre a correção das previsões, mas grande parte das mensagens apresentava baixa confiança.

---

## V2 — Grupos de cinco mensagens

A configuração selecionada, sem clustering e com embeddings, apresentou:

| Métrica     |        V2 |
| ----------- | --------: |
| Accuracy    | **64,0%** |
| Macro F1    | **56,9%** |
| Weighted F1 | **64,0%** |

Melhoria de Macro F1:

```text
V1: 30,2%

V2: 56,9%

+26,7 pontos percentuais
```

A melhoria está associada principalmente à mudança da unidade de classificação: a V2 utiliza grupos de cinco mensagens, fornecendo ao modelo uma quantidade maior de evidência estilística por observação.

---

## Ablação da V2

| Configuração                       |  Macro F1 | Accuracy |
| ---------------------------------- | --------: | -------: |
| V2 com clustering + embeddings     |     55,3% |      62% |
| **V2 sem clustering + embeddings** | **56,9%** |  **64%** |
| V2 sem clustering + sem embeddings |     53,9% |      59% |

A análise indica que:

* o clustering não melhorou o desempenho final;
* os embeddings contribuíram positivamente;
* a configuração selecionada combina TF-IDF, embeddings, características temporais e características comportamentais.

---

# 🎯 Confiança e Seleção de Previsões na V2

A análise de confiança mostrou uma relação clara entre a probabilidade máxima do modelo e a taxa de acerto.

| Faixa de confiança |  Acurácia |         N |
| ------------------ | --------: | --------: |
| ≤ 0,30             |     25,5% |       102 |
| 0,30–0,40          |     28,2% |       280 |
| 0,40–0,50          |     40,2% |       393 |
| 0,50–0,60          |     42,8% |       542 |
| 0,60–0,70          |     55,3% |       461 |
| 0,70–0,80          |     67,4% |       488 |
| 0,80–0,90          | **75,3%** |   **575** |
| 0,90–1,00          | **92,8%** | **1.108** |

A análise de Precisão × Cobertura mostrou aproximadamente:

| Cobertura | Precisão |
| --------: | -------: |
|      ~19% |   ~95,2% |
|    ~42,6% |   ~86,8% |
|    ~61,4% |   ~79,9% |
|    ~80,4% |   ~71,8% |
|    ~97,4% |   ~65,3% |

Esse comportamento motivou a utilização de **P ≥ 0,80 como threshold da regra de negócio**.

O objetivo não é eliminar todos os erros do classificador, mas retirar da camada de intervenção os casos nos quais a evidência estilística é insuficiente.

A partir do threshold:

```text
P < 0,80
    └── silêncio

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
    ├── Agrupamento V2
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
    ├── Regras de confiança
    ├── Cooldown
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

A separação entre os estágios permite observar e controlar independentemente:

1. classificação;
2. geração de texto;
3. delivery.

A LLM não participa da decisão estatística sobre a autoria.

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

---

## 3. Geração da resposta

O segundo worker busca previsões elegíveis.

Somente ocorrências classificadas como:

```text
ERRO_BAIXO
ERRO_ALTO
```

podem gerar uma intervenção.

A LLM recebe apenas o contexto operacional necessário para geração, incluindo:

* categoria;
* personalidade selecionada;
* instruções de geração.

Ela **não recebe o autor real, a probabilidade da previsão ou as features utilizadas pelo classificador**.

A decisão sobre a existência do erro pertence ao classificador e às regras de negócio.

---

## 4. Envio

O terceiro worker busca mensagens geradas e ainda não enviadas.

A mensagem é enviada através da Evolution API e seu status é atualizado no banco.

O delivery é mantido separado da geração para permitir observabilidade independente da etapa de envio.

---

# 🤖 Geração de mensagens

A geração utiliza a API da **Groq**, com o modelo:

```text
openai/gpt-oss-20b
```

O sistema possui sete personalidades:

* ansioso;
* dramático;
* caótico;
* gaúcho;
* mineiro;
* conspiracionista;
* fofo.

As personalidades são combinadas com as categorias produzidas pelo classificador, permitindo diferentes estilos de resposta.

A LLM é responsável **exclusivamente pela geração do texto**.

A decisão de intervir pertence ao pipeline de classificação e às regras de negócio.

---

# 🗄️ Banco de dados

O PostgreSQL armazena o fluxo de processamento.

## `mensagens_raw`

```text
id
membro
data_hora
mensagem
processada
```

Armazena as mensagens recebidas do WhatsApp.

---

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

A tabela possui uma restrição de unicidade por mensagem para evitar previsões duplicadas.

---

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

## `recompensas`

```text
id
membro
recompensa
criado_em
```

Armazena o histórico das recompensas calculadas para o Diário do Inspetor.

---

# 🕵️ Diário do Inspetor

Além da classificação em tempo real, o SCIA possui uma camada de produto chamada **Diário do Inspetor**.

Uma vez por dia, o sistema gera um PDF autônomo utilizando o dossiê produzido naquele período.

Cada edição possui três partes:

```text
Capa
  ↓
Mural de Procurados
  ↓
Dizeres do Inspetor
```

O documento não depende de episódios anteriores nem mantém continuidade narrativa entre os dias.

---

## Mural de Procurados

O mural apresenta os **13 participantes** com cartazes individuais e uma recompensa calculada para cada membro.

A recompensa é baseada nos eventos observados:

* `ERRO_BAIXO` aumenta a recompensa em **500 pontos**;
* `ERRO_ALTO` aumenta a recompensa em **1.000 pontos**;
* `ACERTO_BAIXO` reduz a recompensa em **100 pontos**;
* `ACERTO_ALTO` reduz a recompensa em **250 pontos**.

Após a soma dos eventos, é aplicado um fator de atividade baseado na quantidade de grupos avaliados:

```python
peso_atividade = 1 + math.log1p(grupos_avaliados)
```

O resultado é persistido na tabela `recompensas`.

---

## Dizeres do Inspetor

A narrativa diária é gerada por LLM a partir de um resumo factual dos eventos observados no período.

A LLM não decide quem errou e não participa da classificação.

Sua função é transformar os eventos já determinados pelo pipeline em uma narrativa noir dentro do universo do Inspetor.

A narrativa pode formular hipóteses ou suspeitas baseadas no dossiê, mas a identificação das ocorrências permanece determinada pelo sistema estatístico.

---

## Automação diária

A rotina é executada automaticamente por cron dentro do ambiente Docker:

```text
cron
  ↓
atualização das recompensas
  ↓
consulta das previsões do período
  ↓
geração da narrativa
  ↓
montagem do PDF
  ↓
envio via Evolution API
```

O período diário é tratado como intervalo semiaberto:

```text
[início, fim)
```

Isso evita sobreposição entre dias consecutivos.

---

# 🐳 Docker e infraestrutura

A aplicação é executada utilizando Docker Compose.

A infraestrutura possui:

* Evolution API;
* Redis;
* PostgreSQL;
* webhook FastAPI.

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

---

## Python 3.12

A imagem do webhook utiliza Python 3.12.

O ambiente de desenvolvimento e produção foi padronizado nessa versão devido à compatibilidade dos artefatos utilizados pelo projeto.

---

## PyTorch CPU-only

A inferência do SCIA ocorre em CPU.

A imagem Docker utiliza a distribuição **CPU-only do PyTorch**, evitando o carregamento de bibliotecas CUDA/NVIDIA que não são necessárias para a operação atual.

Essa mudança reduziu significativamente o tamanho das dependências e evitou problemas de armazenamento durante a construção da imagem.

---

## Oracle Cloud

A aplicação está implantada em uma VM da **Oracle Cloud Free Tier**.

O ambiente utiliza:

* Ubuntu 22.04;
* arquitetura x86_64;
* Docker;
* Docker Compose;
* aproximadamente 4 GB de swap;
* PyTorch CPU-only.

A experiência de implantação mostrou que a memória disponível precisa ser considerada principalmente devido ao carregamento dos artefatos de NLP/ML.

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
│   ├── cartazes/
│   │   ├── templates/
│   │   └── ...
│   │
│   ├── diario/
│   │   ├── cron_fim_dia.py
│   │   ├── narrativa.py
│   │   ├── pdf.py
│   │   └── ...
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
├── relatorios/
│   ├── dados.py
│   ├── metricas.py
│   ├── graficos.py
│   ├── pdf.py
│   └── semanal.py
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

| Notebook                       | Descrição                                                     |
| ------------------------------ | ------------------------------------------------------------- |
| `01_eda_processamento.ipynb`   | Análise exploratória e processamento dos dados                |
| `02_embeddings_tfidf.ipynb`    | Geração de embeddings e representações TF-IDF                 |
| `03_clustering.ipynb`          | UMAP e K-Means                                                |
| `04_modeling.ipynb`            | Treinamento e comparação dos modelos                          |
| `05_model_analysis.ipynb`      | Análise detalhada dos resultados offline e erros              |
| `06_prod_model_analysis.ipynb` | Monitoramento contínuo do comportamento do modelo em produção |

---

## `06_prod_model_analysis.ipynb`

O sexto notebook foi criado para transformar a análise de produção em um processo reexecutável.

Ele lê diretamente as tabelas do banco e permite acompanhar a evolução do SCIA conforme novas mensagens são acumuladas.

O notebook está dividido em cinco frentes:

### 1. Saúde do Pipeline e SLAs

* conversão entre `mensagens_raw`, `previsoes` e `mensagens_bot`;
* retenção do pré-processamento;
* intervalos de confiança para proporções;
* latência do Worker ML;
* latência da LLM;
* preparação para medir a latência do Worker 3;
* distribuição dos status.

### 2. Desempenho e confiança do modelo

* Accuracy;
* Macro F1;
* Weighted F1;
* classification report por membro;
* matriz de confusão;
* distribuição das probabilidades;
* teste de Mann-Whitney entre probabilidades de acertos e erros;
* tamanho de efeito rank-biserial.

O teste de Mann-Whitney é utilizado para verificar se previsões corretas tendem a apresentar probabilidades maiores que previsões incorretas.

Essa análise avalia a relação entre **confiança e acerto**, e não constitui uma avaliação formal de calibração probabilística.

### 3. Impostores e Doppelgängers

* taxa geral de acertos e erros;
* frequência de previsões incorretas por autor real;
* identidades mais frequentemente atribuídas pelo modelo;
* distribuição das categorias de intervenção;
* matriz de Doppelgängers;
* identificação dos principais pares de confusão.

### 4. Monitoramento da LLM e Delivery

* quantidade de textos gerados;
* tamanho das respostas;
* frequência das personalidades;
* variabilidade de tamanho por personalidade;
* integridade dos textos;
* status de envio;
* latência da Groq;
* latência do delivery.

### 5. Drift e mudança temporal

* volume de mensagens por horário;
* distribuição de características das mensagens;
* Macro F1 por semana;
* comparação entre janelas temporais;
* baseline da primeira semana;
* testes de mudança de distribuição quando houver múltiplas semanas.

---

# 📈 Resultados em Produção

Os resultados de produção são acompanhados separadamente dos resultados offline.

Isso é importante porque a produção possui condições diferentes do conjunto de teste histórico, além de utilizar regras de confiança e uma unidade de observação baseada em grupos de cinco mensagens.

---

## V1 — Produção

A janela histórica de produção da V1 apresentou:

| Métrica                        |      V1 — Produção |
| ------------------------------ | -----------------: |
| Período analisado              |         **4 dias** |
| Previsões                      |          **1.849** |
| Accuracy                       |         **27,91%** |
| Macro F1                       |         **0,2446** |
| Weighted F1                    |         **0,2927** |
| Mediana de confiança — acertos |         **0,6550** |
| Mediana de confiança — erros   |         **0,4206** |
| Intervenções geradas           |            **124** |
| Mensagens acima do threshold   |         **24,12%** |
| Taxa de intervenção            |          **6,71%** |
| Latência ML                    |  **mediana 6,14s** |
| Latência LLM                   |  **mediana 1,81s** |
| Delivery                       | **mediana 33,86s** |

A V1 funciona como baseline operacional para a evolução do sistema.

---

## V2 — Produção

A V2 já está integrada à pipeline operacional utilizando grupos de cinco mensagens.

Uma janela recente de produção apresentou:

| Métrica                    | V2 — Produção |
| -------------------------- | ------------: |
| Mensagens recebidas        |       **858** |
| Previsões/grupos avaliados |       **164** |
| Accuracy                   |     **57,3%** |
| Macro F1                   |     **48,3%** |
| Weighted F1                |     **57,4%** |
| Cobertura `P ≥ 0,80`       |     **28,7%** |
| Accuracy em `P ≥ 0,80`     |     **93,6%** |
| Macro F1 em `P ≥ 0,80`     |     **94,2%** |
| Weighted F1 em `P ≥ 0,80`  |     **94,0%** |
| Intervenções geradas       |         **3** |
| Delivery das intervenções  |      **100%** |

Esses números representam uma janela observada em produção e não substituem a avaliação offline.

A diferença entre os resultados reforça a necessidade de acompanhamento contínuo do comportamento do modelo.

---

# 📊 Relatório semanal

Além do Diário do Inspetor, o projeto possui uma camada de análise técnica semanal.

O relatório reúne:

* saúde do pipeline;
* SLAs;
* desempenho do modelo;
* distribuição de confiança;
* impostores;
* Doppelgängers;
* comportamento da LLM;
* delivery;
* drift;
* comparação temporal.

A rotina é executada automaticamente através de cron.

---

# 🧪 Operação

A operação do SCIA separa três responsabilidades:

```text
Worker 1
   │
   └── ingestão + processamento + ML

Worker 2
   │
   └── regras + geração LLM

Worker 3
   │
   └── delivery
```

Isso permite acompanhar separadamente:

* distribuição das previsões;
* nível de confiança;
* frequência de intervenções;
* distribuição entre ACERTO e ERRO;
* principais confusões;
* comportamento da LLM;
* latência de processamento;
* sucesso de delivery.

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

A separação entre essas etapas permite que:

* o modelo estatístico seja responsável pela **identificação**;
* a probabilidade seja utilizada para definir a **confiança operacional**;
* as regras de negócio determinem **quando intervir**;
* a LLM seja responsável exclusivamente pela **geração do texto**;
* a Evolution API seja responsável pelo **delivery**.

Na V2, essa separação é complementada por:

* `P ≥ 0,80` como crivo mínimo de confiança;
* categorias `ACERTO_BAIXO`, `ACERTO_ALTO`, `ERRO_BAIXO` e `ERRO_ALTO`;
* cooldown de cinco minutos por pessoa;
* intervenção somente em erros elegíveis.

O Diário do Inspetor adiciona uma camada diferente:

```text
Machine Learning
      ↓
Eventos observados
      ↓
Dossiê diário
      ↓
Narrativa + recompensas
      ↓
Diário do Inspetor
```

Assim, o projeto deixa de ser apenas um classificador de autoria e passa a funcionar como um sistema completo de observação, decisão, intervenção e registro.

---

# 🚀 Próximos passos

* acumular mais semanas de produção para avaliar estabilidade temporal;
* acompanhar a relação entre confiança e acerto;
* acompanhar os principais Doppelgängers;
* monitorar drift e mudanças na distribuição das mensagens;
* avaliar possíveis ajustes no threshold somente após maior volume de dados;
* monitorar continuamente as latências dos três estágios;
* acompanhar o comportamento real do delivery;
* manter o Diário do Inspetor como artefato diário de observabilidade e interação;
* avaliar otimizações adicionais de infraestrutura conforme o volume de mensagens;
* continuar comparando desempenho offline e produção;
* avaliar novas versões do modelo somente após acumular evidência suficiente da V2 em produção.

---

# 👨‍💻 Autor

**Kauã Dias**

Estudante de Estatística — Universidade Federal do Espírito Santo (UFES)

GitHub: https://github.com/kauadp

LinkedIn: https://linkedin.com/in/kauad

---

# 📄 Licença

Projeto desenvolvido para fins de estudo e experimentação em:

* Machine Learning;
* NLP;
* classificação de autoria;
* estilometria;
* sistemas de intervenção automatizada;
* integração com LLMs;
* monitoramento de modelos em produção;
* geração automatizada de relatórios.

Os dados originais utilizados no treinamento não são disponibilizados publicamente por conterem mensagens privadas de um grupo do WhatsApp.
