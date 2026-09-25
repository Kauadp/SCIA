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

O projeto combina **NLP, embeddings, TF-IDF, clustering, aprendizado supervisionado e geração de texto por LLM** para observar mensagens em tempo real e decidir quando uma intervenção do SCIA deve acontecer.

---

# 📖 Sobre o projeto

O **SCIA** nasceu a partir da análise histórica de mensagens de um grupo do WhatsApp, buscando responder a uma pergunta simples:

> **É possível identificar quem escreveu uma mensagem apenas observando a forma como essa pessoa escreve?**

Para investigar essa hipótese, foi construída uma pipeline de Machine Learning utilizando aproximadamente **100 mil mensagens** de **13 participantes**.

O projeto passou por diferentes etapas de processamento e modelagem:

- análise exploratória;
- limpeza e processamento das mensagens;
- geração de embeddings;
- TF-IDF;
- redução de dimensionalidade;
- clustering;
- criação de variáveis temporais;
- treinamento e comparação de modelos;
- análise de erros e confiança;
- integração do modelo a um sistema em tempo real;
- monitoramento do comportamento em produção.

O projeto evoluiu de um experimento offline para um sistema capaz de receber mensagens diretamente do WhatsApp, realizar a previsão e, quando a confiança ultrapassa o limiar definido, gerar uma resposta utilizando uma LLM.

---

# 🎯 Objetivos

- Investigar se padrões linguísticos permitem identificar os autores das mensagens;
- construir um classificador multiclasse para prever o autor de uma mensagem;
- analisar os padrões de comunicação presentes no grupo;
- identificar situações em que o modelo possui maior ou menor confiança;
- criar um sistema capaz de processar mensagens em tempo real;
- gerar intervenções automatizadas utilizando uma LLM;
- observar o comportamento do modelo em produção antes de habilitar o envio automático.

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

Também são utilizadas informações como:

- dia da semana;
- horário;
- quantidade de caracteres;
- quantidade de palavras;
- caracteres por mensagem.

## 2. Representação textual

### Embeddings

Foi utilizado:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

Cada mensagem é representada por um vetor de **384 dimensões**.

### TF-IDF

A representação que apresentou melhor desempenho foi baseada em **n-gramas de caracteres**:

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

O projeto utiliza clustering como uma das fontes de informação do classificador.

O processo utiliza:

- **UMAP** para redução de dimensionalidade;
- **K-Means** para agrupamento;
- **25 clusters** na versão final.

O modelo UMAP final é armazenado em `artifacts/clustering/umap_cluster.joblib`.

### ⚠️ Artefato pesado do UMAP

O artefato final do UMAP possui aproximadamente **458 MB** e, por isso, não é versionado no GitHub.

Durante o deploy da aplicação, o arquivo é transferido **diretamente da máquina de desenvolvimento para a VM** utilizando `scp` e colocado no diretório `artifacts/clustering/` antes da construção da imagem Docker.

Essa decisão evita ultrapassar o limite de tamanho de arquivos do GitHub e mantém o restante dos artefatos versionados normalmente.

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
- early stopping.

---

# 📊 Desempenho offline

O modelo final apresentou aproximadamente:

| Métrica | Teste histórico |
|---|---:|
| Accuracy | 34% |
| Macro F1 | 30,2% |
| Weighted F1 | 35% |

O problema possui **13 classes**, com diferentes níveis de separabilidade entre os participantes.

---

# 🎯 Sistema de confiança

A probabilidade produzida pelo classificador é utilizada para decidir se uma mensagem deve gerar uma intervenção.

| Probabilidade | Classificação |
|---|---|
| `< 0.70` | Sem intervenção |
| `0.70 ≤ p < 0.80` | BAIXO |
| `0.80 ≤ p < 0.90` | MEDIO |
| `0.90 ≤ p ≤ 1.00` | ALTO |

Quando a confiança é inferior a `0.70`, nenhuma mensagem é enviada para a etapa de geração.

As categorias utilizadas internamente são:

```text
ACERTO_BAIXO
ACERTO_MEDIO
ACERTO_ALTO
ERRO_BAIXO
ERRO_MEDIO
ERRO_ALTO
```

As categorias `ACERTO` e `ERRO` são utilizadas durante a análise porque o autor real está disponível no banco de produção. O modelo em si não recebe essa informação para realizar a previsão.

---

# 🧩 Arquitetura

O sistema é dividido em três workers principais:

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

O primeiro worker busca mensagens ainda não processadas.

Para cada mensagem são executadas:

1. extração das características;
2. transformação do texto;
3. identificação do cluster;
4. previsão do autor;
5. cálculo da probabilidade;
6. classificação da confiança.

O resultado é armazenado em `previsoes`.

## 3. Geração da resposta

O segundo worker busca previsões elegíveis.

A LLM recebe:

- categoria;
- personalidade selecionada;
- instruções de geração.

Ela **não recebe o autor real, a probabilidade da previsão ou as features utilizadas pelo classificador**.

Isso mantém a geração separada da lógica de classificação.

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

A LLM é responsável **exclusivamente pela geração do texto**. A decisão de intervir pertence ao pipeline de classificação e regras.

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

A imagem do webhook foi atualizada de **Python 3.11 para Python 3.12**.

A mudança foi necessária porque o artefato `umap_cluster.joblib` apresentou incompatibilidade ao ser carregado no ambiente Python 3.11, apesar das versões das principais bibliotecas serem equivalentes. O artefato havia sido gerado no ambiente Python 3.12.

A imagem atual utiliza:

```dockerfile
FROM python:3.12-slim
```

Com Python 3.12, o carregamento do UMAP foi normalizado e o container passou a operar sem o ciclo de reinicialização observado anteriormente.

## Desempenho observado do container

Na execução local atual, o consumo de memória observado foi aproximadamente:

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

### Oracle Cloud

A arquitetura foi construída para poder ser executada posteriormente em uma VM. Entretanto, o consumo observado indica que uma máquina com apenas **1 GB de RAM não é confortável para a versão atual** do SCIA.

A estratégia atual é continuar validando o comportamento do sistema localmente antes de realizar o deploy definitivo na VM.

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

Essa análise permite estudar a estrutura de sobreposição entre os estilos aprendidos pelo classificador.

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
- preparação para testes de mudança de distribuição quando houver múltiplas semanas.

Com apenas uma semana de dados, o notebook registra a janela atual como **baseline de produção** e não realiza inferência de drift temporal.

---

# 📈 Resultados atuais em produção

A primeira janela consolidada de produção analisada contém **338 previsões**.

### Desempenho do classificador

| Métrica | Produção |
|---|---:|
| Accuracy | **27,81%** |
| Macro F1 | **0,2270** |
| Weighted F1 | **0,2985** |

O desempenho observado em produção está abaixo do desempenho do teste histórico, mas a primeira janela ainda é insuficiente para determinar se essa diferença representa uma degradação estrutural ou apenas variação da amostra.

O desempenho também é heterogêneo entre os participantes. Na amostra atual, alguns membros apresentam F1 mais elevado que outros, indicando diferentes níveis de separabilidade entre os estilos.

### Confiança das previsões

Na amostra atual:

- **94 previsões corretas**;
- **244 previsões incorretas**;
- mediana da probabilidade nos acertos: **0,6113**;
- mediana da probabilidade nos erros: **0,4627**;
- Mann-Whitney unilateral: `U = 14807`;
- `p = 1,68 × 10⁻⁵`;
- rank-biserial aproximado: **+0,291**.

Os resultados fornecem evidência de que a confiança do modelo contém informação sobre a correção da previsão.

### Intervenções

Foram gerados **89 textos** a partir das previsões elegíveis, correspondendo a aproximadamente **26,3% das previsões**.

Entre essas intervenções:

- **39** correspondiam a `ACERTO`;
- **50** correspondiam a `ERRO`.

A distribuição observada foi:

```text
ERRO_BAIXO    24
ACERTO_ALTO   20
ERRO_MEDIO    14
ERRO_ALTO     12
ACERTO_BAIXO  10
ACERTO_MEDIO   9
```

A análise de Doppelgängers mostrou que os erros não são distribuídos uniformemente entre as identidades. Algumas identidades aparecem frequentemente como destinos das previsões incorretas, enquanto determinados pares apresentam padrões recorrentes de confusão.

### LLM

Na janela analisada:

- **89 textos gerados**;
- nenhum texto nulo ou vazio;
- mediana de aproximadamente **77 palavras** por resposta;
- latência mediana da Groq de aproximadamente **0,83 s**.

O Worker 3 permanece desativado, portanto os registros continuam como `PENDENTE_ENVIO` e ainda não existem métricas reais de entrega no WhatsApp.

### O que ainda não pode ser concluído

A primeira janela de produção permite observar o comportamento inicial do sistema, mas ainda não permite concluir:

- existência de drift temporal;
- estabilidade de longo prazo do Macro F1;
- necessidade de alterar o limiar `0.70`;
- estabilidade da taxa de intervenção;
- persistência dos mesmos Doppelgängers ao longo do tempo.

A estratégia atual é acumular novas observações e reexecutar o `06_prod_model_analysis.ipynb` periodicamente.

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
│      Machine Learning │
│                       │
│ Quem escreveu isso?   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│      Confiança        │
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

- acumular mais dados de produção;
- reexecutar periodicamente o `06_prod_model_analysis.ipynb`;
- comparar novas janelas com o baseline atual;
- acompanhar os principais Doppelgängers;
- verificar a estabilidade da relação entre confiança e acerto;
- avaliar possíveis ajustes nos limites de confiança somente após maior volume de dados;
- habilitar o Worker 3 após a fase de observação;
- monitorar latência e sucesso real de delivery;
- avaliar melhorias futuras no classificador caso os padrões observados em produção sejam persistentes;
- avaliar o deploy definitivo em VM após validar os requisitos computacionais.

---

# 👨‍💻 Autor

**Kauã Dias**  
Estudante de Estatística — Universidade Federal do Espírito Santo (UFES)

GitHub: https://github.com/kauadp  
LinkedIn: https://linkedin.com/in/kauad

---

# 📄 Licença

Projeto desenvolvido para fins de estudo e experimentação em **Machine Learning, NLP, sistemas de recomendação de intervenção e integração com LLMs**.

Os dados originais utilizados no treinamento não são disponibilizados publicamente por conterem mensagens privadas de um grupo do WhatsApp.
