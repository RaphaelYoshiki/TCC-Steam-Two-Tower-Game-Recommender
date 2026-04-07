# Game Recommender System

Um sistema inteligente de recomendação de jogos baseado em machine learning com backend em FastAPI e interface gráfica em Toga. O sistema utiliza um modelo Two-Tower para gerar recomendações personalizadas com suporte a chat via integração com DeepSeek.

## Características

- **Modelo Two-Tower Neural**: Arquitetura de aprendizado profundo para recomendações personalizadas
- **Backend FastAPI**: API REST robusta para servir recomendações
- **Interface GUI**: Aplicação desktop com Toga para interação intuitiva
- **Chat Inteligente**: Integração com DeepSeek para recomendações conversacionais
- **Pipeline de Dados**: Sistema completo de extração, limpeza e processamento de dados
- **Containerização**: Suporte para Docker para deployment facilitado

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                   GUI (Toga App)                        │
│          - Chat Interface                               │
│          - Interaction with Users                       │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│           DeepSeek Client (LLM Integration)             │
│          - Conversation Management                      │
│          - Tool Calling                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend                            │
│          - /recommend Endpoint                          │
│          - Game Lookup Service                          │
│          - Model Inference                              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│        Two-Tower Model (TensorFlow/Keras)              │
│          - User Tower (Embedding Layer)                 │
│          - Game Tower (Embedding Layer)                 │
│          - Similarity Scoring                           │
└─────────────────────────────────────────────────────────┘
```

## Estrutura do Projeto

```
project_root/
├── scripts/                          # Scripts de treinamento e preparação
│   ├── requirements.txt              # Dependências do projeto
│   ├── data_prep.py                  # Preparação de dados
│   ├── two_tower_model.py            # Definição do modelo
│   └── two_tower_training_NOCHUNK.py # Script de treinamento
│
├── datamining/                       # Pipeline de extração e processamento
│   ├── game_data_aquisition.py       # Coleta de dados de jogos
│   ├── game_data_manip.py            # Manipulação de dados de jogos
│   ├── user_interaction_aquisition.py# Coleta de interações
│   ├── interaction_data_manip.py     # Processamento de interações
│   ├── interactions_subsampler.py    # Subamostragem de dados
│   ├── gamedata_value_lists.py       # Listas de valores únicos
│   └── utilitary_codes/              # Funções utilitárias
│       ├── list_normalizer.py
│       ├── outlier_removal.py
│       ├── prefilter.py
│       ├── preprocesser.py
│       └── unique_value_lister.py
│
├── recommendation_backend/           # Backend da API
│   ├── app_backend.py                # Aplicação FastAPI
│   ├── model_loader.py               # Carregamento de modelos
│   ├── recommend.py                  # Lógica de recomendação
│   ├── game_lookup.py                # Busca de informações de jogos
│   ├── schemas.py                    # Schemas Pydantic
│   └── aux_files/
│       ├── id_mappings.json          # Mapeamento de IDs
│       └── lang_id_map.json          # Mapa de idiomas
│
├── recommendation_gui/               # Interface Gráfica
│   ├── gui_app.py                    # Aplicação principal
│   ├── backend_client.py             # Cliente do backend
│   ├── deepseek_client.py            # Cliente DeepSeek
│   ├── chat_state.py                 # Gerenciamento de estado
│   ├── prompts.py                    # Prompts para LLM
│   └── tools.py                      # Ferramentas disponíveis
│
├── models/                           # Modelos treinados (.keras)
│   └── two_tower_realusers_*.keras
│
├── pickle_files/                     # Artefatos serializados
│   ├── game_map.pkl
│   └── game_max_lengths.pkl
│
├── csv_files/                        # Dados em formato CSV
│   ├── cleaned_interactions_shuffled.csv
│   └── treated_dataframe.csv
│
├── model_plots/                      # Gráficos de treinamento
│   └── history_*.txt
│
├── Dockerfile.txt                    # Configuração Docker
└── __init__.py
```

## Instalação

### Pré-requisitos

- Python 3.9+
- pip ou conda
- TensorFlow 2.15.0
- Docker (opcional, para containerização)

### Setup Local

1. **Clone o repositório:**
```bash
cd project_root
```

2. **Crie e ative um ambiente virtual:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r scripts/requirements.txt
```

### Estrutura de Dependências

O projeto utiliza as seguintes bibliotecas principais:

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| TensorFlow | 2.15.0 | Framework de ML profundo |
| tensorflow-recommenders | 0.7.3 | Modelos de recomendação |
| tensorflow-ranking | 0.5.5 | Ranking neural |
| NumPy | 1.26.4 | Computação numérica |
| Pandas | 2.3.0 | Manipulação de dados |
| Matplotlib | 3.9.4 | Visualização |
| Requests | - | Requisições HTTP |
| FastAPI | - | API REST |
| Uvicorn | - | Servidor ASGI |
| Toga | - | Interface gráfica |
| Pydantic | - | Validação de dados |
| SciPy | - | Estatística e ciência de dados |
| scikit-learn | - | Pré-processamento e ML auxiliar |
| dateparser | - | Parsing de datas |
| torch | - | Aprendizado de máquina (PyTorch) |
| torch-geometric | - | Processamento de grafos |

## Uso

### Fluxo de Execução Completo

Para executar o sistema completo, siga estes passos em ordem:

1. **Coleta e filtração de dados**
2. **Preparação de dados para treino**
3. **Treinamento do modelo**
4. **Execução do backend e da GUI**

#### ⚠️ Importante sobre Coleta de Dados

**A API do Steam foi alterada e não é mais acessível publicamente.**

- **API antiga**: `https://api.steampowered.com/ISteamApps/GetAppList/v2/`
- **API nova**: `https://partner.steamgames.com/doc/webapi/IStoreService#GetAppList`

A nova API **não é de acesso público** e exige uma conta de parceiro Steamworks com permissões de administrador. Portanto, a coleta de dados não pode ser realizada livremente atualmente.

**O caminho recomendado hoje é usar o arquivo pré-processado:**
- Baixe `cleaned_interactions_shuffled.csv` do dataset Kaggle em `https://www.kaggle.com/datasets/rkyoshiki/steam-dataset`
- Copie o arquivo para `csv_files/`
- Se necessário, renomeie o arquivo para `cleaned_interactions_shuffled.csv`

### Passo a Passo de Execução (Usando Dados Pré-coletados)

#### **Passo 0: Baixe o arquivo pré-processado**
```bash
# Coloque o arquivo baixado em csv_files/
# Se necessário, ajuste o caminho e o nome do arquivo
cp /caminho/para/cleaned_interactions_shuffled.csv csv_files/cleaned_interactions_shuffled.csv
```

No Windows:
```powershell
Copy-Item .\download\cleaned_interactions_shuffled.csv .\csv_files\cleaned_interactions_shuffled.csv
```

#### **Passo 1: Entenda o pipeline de coleta e filtragem**

Se você tiver acesso à API do Steam ou quiser reproduzir o processo de preparação, execute:
```bash
cd datamining
python interaction_data_manip.py
python interactions_subsampler.py
```

> Observação: devido às limitações atuais da API Steam, este passo é opcional para execução prática. O arquivo `cleaned_interactions_shuffled.csv` do Kaggle já contém interações pré-processadas.

#### **Passo 2: Preparação para Treinamento**
```bash
cd ../scripts
python data_prep.py
```

#### **Passo 3: Treinamento do Modelo**
```bash
python two_tower_training.py
```

#### **Passo 4: Execução da Aplicação**

**4.1 Backend API**
```bash
cd ../recommendation_backend
python -m uvicorn app_backend:app --reload --host 0.0.0.0 --port 8000
```

**4.2 Interface GUI (em outro terminal)**
```bash
cd ../recommendation_gui
python gui_app.py
```

### Hiperparâmetros Importantes

Os parâmetros principais estão definidos em `scripts/two_tower_training.py` e influenciam o comportamento do treinamento:

- `BATCH_SIZE`: tamanho do lote de treino. Valores maiores treinam mais exemplos por atualização de gradiente, mas exigem mais memória.
- `EMBEDDING_DIM`: dimensão das representações latentes do usuário e do jogo. Valores maiores aumentam a capacidade do modelo, mas também aumentam o custo computacional e o risco de overfitting.
- `EPOCHS`: número de épocas de treinamento. Mais épocas podem melhorar a convergência, mas também podem levar a overfitting se o modelo for treinado demais.
- `NEG_K`: número de exemplos negativos gerados por usuário. Aumentar esse valor deixa o ranking mais desafiador e pode melhorar a qualidade da recomendação, mas deixa o treino mais lento.
- `VAL_SPLIT`: proporção de usuários reservada para validação. Um valor maior dá uma avaliação mais confiável, porém reduz a quantidade de dados usados no treino.
- `TOP_N`: número de interações positivas usadas por usuário. Ajusta a riqueza de sinais positivos no conjunto de treino.

### 📋 Detalhes dos Scripts Executados

| Script | Função | Arquivos de Entrada | Arquivos de Saída |
|--------|--------|---------------------|-------------------|
| `interaction_data_manip.py` | Limpa e processa interações | `LLM/utils/all_data/interactions.csv` e `LLM/utils/all_data/treated_dataframe.csv` | `LLM/utils/all_data/cleaned_interactions_shuffled.csv` |
| `interactions_subsampler.py` | Divide em treino/validação | `LLM/utils/all_data/cleaned_interactions_shuffled.csv` | `LLM/utils/all_data/interactions_train.csv`, `LLM/utils/all_data/interactions_val.csv` |
| `data_prep.py` | Prepara dados para ML | `csv_files/treated_dataframe.csv`, `csv_files/cleaned_interactions_shuffled.csv` | `pickle_files/*` |
| `two_tower_training_NOCHUNK.py` | Treina o modelo | Dados preparados | `models/two_tower_realusers_*.keras` |

### 1. Backend API

**Pré-requisitos:** Modelo treinado disponível em `models/`

**Inicie o servidor FastAPI:**
```bash
cd recommendation_backend
python -m uvicorn app_backend:app --reload --host 0.0.0.0 --port 8000
```

**Endpoint disponível:**
```
POST /recommend
Content-Type: application/json

{
  "lang_id": 0,
  "top_genre": 15,
  "top_cat": 2,
  "top_tag": 45,
  "genre_dominance": 0.65,
  "genre_diversity": 0.40
}
```

**Resposta:**
```json
{
  "recommendations": [
    {
      "id": 570,
      "name": "Game Name",
      "score": 0.95
    },
    ...
  ]
}
```

### 2. Interface GUI

**Pré-requisitos:** Backend API rodando e modelo treinado disponível

**Inicie a aplicação desktop:**
```bash
cd recommendation_gui
python gui_app.py
```

A aplicação oferece:
- Interface de chat conversacional
- Recomendações de jogos em tempo real
- Integração com DeepSeek para interações naturais

## Treinamento de Modelos

### Preparação de Dados (Usando Dados Pré-coletados)
```bash
cd project_root

# Coloque o arquivo pré-processado em csv_files/
# Se você já tiver cleaned_interactions_shuffled.csv, não precisa rodar a coleta.
cd datamining
python interaction_data_manip.py        # Reproduz o pipeline de limpeza, quando disponível
python interactions_subsampler.py       # Divide em treino/validação
```

### Treinamento
```bash
cd ../scripts
python data_prep.py
python two_tower_training_NOCHUNK.py
```

Os modelos treinados são salvos em `models/two_tower_realusers_*.keras`.

## Modelos Disponíveis

O projeto inclui múltiplas versões de modelo treinado:
- `two_tower_realusers_12.keras` até `two_tower_realusers_27.keras`
- Versões anteriores em `VM_outputs/models/`

O modelo padrão utilizado é: `two_tower_realusers_26.keras`

## Pipeline de Dados

### ⚠️ Limitações da API do Steam

**Importante:** A coleta de dados atualmente enfrenta limitações devido a mudanças na API do Steam.

- **API antiga (descontinuada)**: `https://api.steampowered.com/ISteamApps/GetAppList/v2/`
- **API nova (restrita)**: `https://partner.steamgames.com/doc/webapi/IStoreService#GetAppList`

A nova API **não é pública** e requer:
- Conta de parceiro Steamworks
- Permissões de administrador
- Chave de API especial

**Status atual:** A coleta de dados não pode ser realizada livremente. Use os dados pré-coletados disponíveis no projeto.

### Fluxo de Processamento

1. **Aquisição**: Coleta de dados de jogos e interações de usuários (atualmente limitada)
2. **Limpeza**: Remoção de outliers e normalização de dados
3. **Filtragem**: Pré-filtragem e processamento
4. **Transformação**: Conversão para formato adequado para o modelo
5. **Subamostragem**: Redução balanceada do dataset

### Arquivos de Dados

- `csv_files/cleaned_interactions_shuffled.csv` - Interações processadas
- `csv_files/treated_dataframe.csv` - Dados tratados
- `pickle_files/game_map.pkl` - Mapeamento de IDs de jogos
- `pickle_files/game_max_lengths.pkl` - Comprimentos máximos de sequências

## Arquitetura do Modelo Two-Tower

### Torre do Usuário
- Entrada: Perfil do usuário (idioma, gênero favorito, categoria, tag, dominância, diversidade)
- Embedding layers para cada atributo
- Camadas densas com regularização L2
- Saída: Representação latente do usuário

### Torre do Jogo
- Entrada: Atributos do jogo (gêneros, categorias, tags, review score)
- Embedding com média ponderada de sequências
- Camadas densas com regularização
- Saída: Representação latente do jogo

### Scoring
- Dot product entre representações
- Top-K retrieval para recomendações finais

## Licença

Projeto acadêmico - TCC (Trabalho de Conclusão de Curso)

---

**Última atualização**: Abril 2026

---

# Game Recommender System (English Version)

An intelligent game recommendation system based on machine learning with FastAPI backend and Toga graphical interface. The system uses a Two-Tower model to generate personalized recommendations with chat support through DeepSeek integration.

## Features

- **Two-Tower Neural Model**: Deep learning architecture for personalized recommendations
- **FastAPI Backend**: Robust REST API for serving recommendations
- **GUI Interface**: Desktop application with Toga for intuitive interaction
- **Intelligent Chat**: DeepSeek integration for conversational recommendations
- **Data Pipeline**: Complete system for data extraction, cleaning, and processing
- **Containerization**: Docker support for easy deployment

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GUI (Toga App)                        │
│          - Chat Interface                               │
│          - User Interaction                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│           DeepSeek Client (LLM Integration)             │
│          - Conversation Management                      │
│          - Tool Calling                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend                            │
│          - /recommend Endpoint                          │
│          - Game Lookup Service                          │
│          - Model Inference                              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│        Two-Tower Model (TensorFlow/Keras)              │
│          - User Tower (Embedding Layer)                 │
│          - Game Tower (Embedding Layer)                 │
│          - Similarity Scoring                           │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
project_root/
├── scripts/                          # Training and preparation scripts
│   ├── requirements.txt              # Project dependencies
│   ├── data_prep.py                  # Data preparation
│   ├── two_tower_model.py            # Model definition
│   └── two_tower_training_NOCHUNK.py # Training script
│
├── datamining/                       # Data extraction and processing pipeline
│   ├── game_data_aquisition.py       # Game data collection
│   ├── game_data_manip.py            # Game data manipulation
│   ├── user_interaction_aquisition.py# User interaction collection
│   ├── interaction_data_manip.py     # Interaction processing
│   ├── interactions_subsampler.py    # Data subsampling
│   ├── gamedata_value_lists.py       # Unique value lists
│   └── utilitary_codes/              # Utility functions
│       ├── list_normalizer.py
│       ├── outlier_removal.py
│       ├── prefilter.py
│       ├── preprocesser.py
│       └── unique_value_lister.py
│
├── recommendation_backend/           # API Backend
│   ├── app_backend.py                # FastAPI application
│   ├── model_loader.py               # Model loading
│   ├── recommend.py                  # Recommendation logic
│   ├── game_lookup.py                # Game information lookup
│   ├── schemas.py                    # Pydantic schemas
│   └── aux_files/
│       ├── id_mappings.json          # ID mappings
│       └── lang_id_map.json          # Language map
│
├── recommendation_gui/               # Graphical Interface
│   ├── gui_app.py                    # Main application
│   ├── backend_client.py             # Backend client
│   ├── deepseek_client.py            # DeepSeek client
│   ├── chat_state.py                 # State management
│   ├── prompts.py                    # LLM prompts
│   └── tools.py                      # Available tools
│
├── models/                           # Trained models (.keras)
│   └── two_tower_realusers_*.keras
│
├── pickle_files/                     # Serialized artifacts
│   ├── game_map.pkl
│   └── game_max_lengths.pkl
│
├── csv_files/                        # CSV data files
│   ├── cleaned_interactions_shuffled.csv
│   └── treated_dataframe.csv
│
├── model_plots/                      # Training plots
│   └── history_*.txt
│
├── Dockerfile.txt                    # Docker configuration
└── __init__.py
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda
- TensorFlow 2.15.0
- Docker (optional, for containerization)

### Local Setup

1. **Clone the repository:**
```bash
cd project_root
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r scripts/requirements.txt
```

### Dependencies Structure

The project uses the following main libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.15.0 | Deep ML framework |
| tensorflow-recommenders | 0.7.3 | Recommendation models |
| tensorflow-ranking | 0.5.5 | Neural ranking |
| NumPy | 1.26.4 | Numerical computing |
| Pandas | 2.3.0 | Data manipulation |
| Matplotlib | 3.9.4 | Visualization |
| Requests | - | HTTP requests |
| FastAPI | - | REST API |
| Uvicorn | - | ASGI server |
| Toga | - | GUI framework |
| Pydantic | - | Data validation |
| SciPy | - | Statistical and scientific tools |
| scikit-learn | - | Preprocessing and ML utilities |
| dateparser | - | Date parsing |
| torch | - | Machine learning (PyTorch) |
| torch-geometric | - | Graph processing |

## Usage

### Complete Execution Flow

To run the complete system, follow these steps in order:

1. **Data collection and filtering**
2. **Training data preparation**
3. **Model training**
4. **Run backend and GUI**

#### ⚠️ Important about Data Collection

**The Steam API has been changed and is no longer publicly accessible.**

- **Old API**: `https://api.steampowered.com/ISteamApps/GetAppList/v2/`
- **New API**: `https://partner.steamgames.com/doc/webapi/IStoreService#GetAppList`

The new API **is not public** and requires a Steamworks partner account with administrator permissions. Therefore, data collection cannot be freely performed currently.

**The recommended approach today is to use the preprocessed file:**
- Download `cleaned_interactions_shuffled.csv` from `https://www.kaggle.com/datasets/rkyoshiki/steam-dataset`
- Place it in `csv_files/`
- Rename it to `cleaned_interactions_shuffled.csv` if needed

### Step-by-Step Execution (Using Pre-collected Data)

#### Step 0: Place the preprocessed file
```bash
# Put the downloaded file into csv_files/
cp /path/to/cleaned_interactions_shuffled.csv csv_files/cleaned_interactions_shuffled.csv
```

On Windows:
```powershell
Copy-Item .\download\cleaned_interactions_shuffled.csv .\csv_files\cleaned_interactions_shuffled.csv
```

#### Step 1: Optional pipeline reproduction

If you have Steam API access or want to reproduce the data preparation pipeline, run:
```bash
cd datamining
python interaction_data_manip.py
python interactions_subsampler.py
```

> Note: because Steam API access is currently restricted, this step is optional for practical execution. The Kaggle file already contains preprocessed interactions.

#### Step 2: Training Preparation
```bash
cd ../scripts
python data_prep.py
```

#### Step 3: Model Training
```bash
python two_tower_training.py
```

#### Step 4: Application Execution

**4.1 Backend API**
```bash
cd ../recommendation_backend
python -m uvicorn app_backend:app --reload --host 0.0.0.0 --port 8000
```

**4.2 GUI Interface (in another terminal)**
```bash
cd ../recommendation_gui
python gui_app.py
```

### Hyperparameters

The main training hyperparameters are defined in `scripts/two_tower_training.py` and affect model behavior as follows:

- `BATCH_SIZE`: number of examples per gradient update. Larger values speed up training per epoch but use more memory.
- `EMBEDDING_DIM`: size of latent vectors for user and game representations. Larger values increase capacity but also compute cost and overfitting risk.
- `EPOCHS`: number of training epochs. More epochs may improve performance, but too many can lead to overfitting.
- `NEG_K`: number of negative samples per user. Increasing it makes ranking harder and more robust, but slows training.
- `VAL_SPLIT`: fraction of users reserved for validation. More validation data improves evaluation reliability but reduces training data.
- `TOP_N`: number of positive interactions per user included in the training list. Affects the richness of positive signal and dataset size.

### Script Details

| Script | Function | Input Files | Output Files |
|--------|----------|-------------|--------------|
| `interaction_data_manip.py` | Cleans and processes interactions | `csv_files/treated_dataframe.csv` | `csv_files/cleaned_interactions_shuffled.csv` |
| `interactions_subsampler.py` | Splits into train/validation | `csv_files/cleaned_interactions_shuffled.csv` | `csv_files/interactions_train.csv`, `csv_files/interactions_val.csv` |
| `data_prep.py` | Prepares data for ML | `csv_files/treated_dataframe.csv`, `csv_files/cleaned_interactions_shuffled.csv` | `pickle_files/*` |
| `two_tower_training_NOCHUNK.py` | Trains the model | Prepared data | `models/two_tower_realusers_*.keras` |

### 1. Backend API

**Prerequisites:** Trained model available in `models/`

**Start the FastAPI server:**
```bash
cd recommendation_backend
python -m uvicorn app_backend:app --reload --host 0.0.0.0 --port 8000
```

**Available endpoint:**
```
POST /recommend
Content-Type: application/json

{
  "lang_id": 0,
  "top_genre": 15,
  "top_cat": 2,
  "top_tag": 45,
  "genre_dominance": 0.65,
  "genre_diversity": 0.40
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "id": 570,
      "name": "Game Name",
      "score": 0.95
    },
    ...
  ]
}
```

### 2. GUI Interface

**Prerequisites:** Backend API running and trained model available

**Start the desktop application:**
```bash
cd ../recommendation_gui
python gui_app.py
```

The application provides:
- Conversational chat interface
- Real-time game recommendations
- DeepSeek integration for natural interactions

## Model Training

### Data Preparation (Using Pre-collected Data)
```bash
cd project_root

# Place the preprocessed file in csv_files/
# If you already have cleaned_interactions_shuffled.csv, you can skip data collection.
cd datamining
python interaction_data_manip.py        # Reproduce the cleaning pipeline when available
python interactions_subsampler.py       # Split into train/validation
```

### Training
```bash
cd ../scripts
python data_prep.py
python two_tower_training_NOCHUNK.py
```

Trained models are saved in `models/two_tower_realusers_*.keras`.

## Available Models

The project includes multiple trained model versions:
- `two_tower_realusers_12.keras` to `two_tower_realusers_27.keras`
- Previous versions in `VM_outputs/models/`

The default model used is: `two_tower_realusers_26.keras`

## Data Pipeline

### ⚠️ Steam API Limitations

**Important:** Data collection currently faces limitations due to Steam API changes.

- **Old API (discontinued)**: `https://api.steampowered.com/ISteamApps/GetAppList/v2/`
- **New API (restricted)**: `https://partner.steamgames.com/doc/webapi/IStoreService#GetAppList`

The new API **is not public** and requires:
- Steamworks partner account
- Administrator permissions
- Special API key

**Current status:** Data collection cannot be performed freely. Use the pre-collected data available in the project.

### Processing Flow

1. **Acquisition**: Collection of game data and user interactions (currently limited)
2. **Cleaning**: Outlier removal and data normalization
3. **Filtering**: Pre-filtering and processing
4. **Transformation**: Conversion to model-appropriate format
5. **Subsampling**: Balanced dataset reduction

### Data Files

- `csv_files/cleaned_interactions_shuffled.csv` - Processed interactions
- `csv_files/treated_dataframe.csv` - Treated data
- `pickle_files/game_map.pkl` - Game ID mapping
- `pickle_files/game_max_lengths.pkl` - Maximum sequence lengths

## Two-Tower Model Architecture

### User Tower
- Input: User profile (language, favorite genre, category, tag, dominance, diversity)
- Embedding layers for each attribute
- Dense layers with L2 regularization
- Output: User latent representation

### Game Tower
- Input: Game attributes (genres, categories, tags, review score)
- Embedding with weighted average of sequences
- Dense layers with regularization
- Output: Game latent representation

### Scoring
- Dot product between representations
- Top-K retrieval for final recommendations

## License

Academic project - TCC (Final Course Project)

---

**Last update**: April 2026
