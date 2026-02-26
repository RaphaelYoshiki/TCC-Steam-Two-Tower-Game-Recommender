# two_tower_training_NOCHUNK.py — versão atualizada para modelo Two-Tower com dados reais de interação de usuário
import os
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import pandas as pd
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from matplotlib import pyplot as plt

# ---------------- CONFIGURAÇÕES ----------------
TRAIN_NUM = 27
INTERACTIONS_FILE = './csv_files/cleaned_interactions_shuffled.csv'
GAMES_CSV = './csv_files/treated_dataframe.csv'
BATCH_SIZE = 512
EMBEDDING_DIM = 512
EPOCHS = 50
MODEL_SAVE_PATH = f'./models/two_tower_realusers_{TRAIN_NUM}.keras'
LOSS_PLOT = f'./model_plots/loss_plot_{TRAIN_NUM}.png'
MAP_PLOT = f'./model_plots/map_plot_{TRAIN_NUM}.png'
NDCG_PLOT = f'./model_plots/ndcg_plot_{TRAIN_NUM}.png'
RANKING_METRICS_PLOT = f'./model_plots/ranking_metrics_plot_{TRAIN_NUM}.png'
ROC_PLOT = f'./model_plots/roc_curve_{TRAIN_NUM}.png'
HISTORY_LOG = f'./model_plots/history_{TRAIN_NUM}.txt'
NEG_K = 7
VAL_SPLIT = 0.2
TOP_N = 8
PICKLE_DIR = './pickle_files'
GAME_MAP_PATH = f'{PICKLE_DIR}/game_map.pkl'
USER_MAP_PATH = f'{PICKLE_DIR}/user_map.pkl'
UNIQUE_LANGS_PATH = f'{PICKLE_DIR}/unique_langs.pkl'
GAME_MAX_LENGTHS_PATH = f'{PICKLE_DIR}/game_max_lengths.pkl'
UNIQUE_IDS_PATH = f'{PICKLE_DIR}/unique_ids.pkl'
# ------------------------------------------------

# Importações atualizadas
from data_prep import (
    load_games_dataframe, 
    load_interactions_dataframe, 
    build_game_map, 
    build_user_map, 
    build_user_lists, 
    create_tf_dataset
)
from two_tower_model import create_two_tower_model

# DEBUG
# ------------------------------------------------
def check_data_leakage(train_interactions, test_interactions, user_map, game_map):
    """Verificar se há vazamento entre treino e teste"""
    
    # Verificar sobreposição de usuários
    train_users = set(train_interactions['steamid'].unique())
    test_users = set(test_interactions['steamid'].unique())
    overlap_users = train_users & test_users
    print(f"🔍 Usuários em treino e teste: {len(overlap_users)}")
    
    # Verificar se há jogos de teste no treino
    train_games = set(train_interactions['appid'].unique())
    test_games = set(test_interactions['appid'].unique())
    overlap_games = train_games & test_games
    print(f"🔍 Jogos em treino e teste: {len(overlap_games)}")
    
    # Verificar temporal leakage
    if 'date' in train_interactions.columns:
        max_train_date = train_interactions['date'].max()
        min_test_date = test_interactions['date'].min()
        print(f"🔍 Datas - Treino até: {max_train_date}, Teste desde: {min_test_date}")
        
def debug_labels(user_lists, name="Dataset"):
    """Debug das labels para verificar distribuição"""
    all_labels = []
    for _, _, labels in user_lists:
        all_labels.extend(labels)
    
    all_labels = np.array(all_labels)
    print(f"\n🔍 {name} Label Analysis:")
    print(f"   Total samples: {len(all_labels)}")
    print(f"   Positive labels: {np.sum(all_labels)}")
    print(f"   Negative labels: {len(all_labels) - np.sum(all_labels)}")
    print(f"   Positive ratio: {np.mean(all_labels):.4f}")
    print(f"   Label values: {np.unique(all_labels)}")
    
def debug_negative_sampling(user_lists):
    """Verificar se o negative sampling está correto"""
    positive_counts = []
    negative_counts = []
    list_lengths = []
    
    for _, _, labels in user_lists:
        positive_counts.append(np.sum(labels))
        negative_counts.append(len(labels) - np.sum(labels))
        list_lengths.append(len(labels))
    
    print(f"📊 Estatísticas das Listas:")
    print(f"   Média de positivos por lista: {np.mean(positive_counts):.2f}")
    print(f"   Média de negativos por lista: {np.mean(negative_counts):.2f}")
    print(f"   Comprimento médio da lista: {np.mean(list_lengths):.2f}")
    print(f"   Proporção de positivos: {np.mean(positive_counts) / np.mean(list_lengths):.3f}")
    
def simple_sanity_check(model, user_map, game_map, game_max_lengths):
    """Teste simples para verificar se o modelo está funcionando"""
    print("\n🧪 Sanity Check do Modelo:")
    
    # Pegar um exemplo aleatório
    test_steamid = list(user_map.keys())[0]
    user_data = user_map[test_steamid]
    
    # Criar inputs dummy - ATUALIZADO para novas features
    list_size = 10
    dummy_inputs = {
        'user_lang_id': np.array([user_data.get('lang_id', 0)] * list_size),
        'user_top_genre': np.array([user_data.get('top_genre', 0)] * list_size),
        'user_top_cat': np.array([user_data.get('top_cat', 0)] * list_size),
        'user_top_tag': np.array([user_data.get('top_tag', 0)] * list_size),
        'user_genre_dominance': np.array([[user_data.get('genre_dominance', 0.0)]] * list_size),
        'user_genre_diversity': np.array([[user_data.get('genre_diversity', 0.0)]] * list_size),
        'game_genre_ids': np.random.randint(0, 10, (list_size, game_max_lengths['genre_ids'])),
        'game_category_ids': np.random.randint(0, 10, (list_size, game_max_lengths['category_ids'])),
        'game_user_tag_ids': np.random.randint(0, 10, (list_size, game_max_lengths['user_tag_ids'])),
        'game_review_score': np.random.random((list_size, 1))
    }
    
    # Adicionar batch dimension
    for key in dummy_inputs:
        dummy_inputs[key] = np.expand_dims(dummy_inputs[key], 0)
    
    # Fazer predição
    try:
        prediction = model.predict(dummy_inputs, verbose=0)
        print(f"✅ Modelo funciona - Shape da saída: {prediction['score'].shape}")
        print(f"✅ Valores de saída: {prediction['score'][0]}")
    except Exception as e:
        print(f"❌ Erro no modelo: {e}")

# ------------------------------------------------
# Custom Classes
# ------------------------------------------------    
class RankingMetrics(tf.keras.callbacks.Callback):
    def __init__(self, test_data, user_map, game_map, game_max_lengths, top_k_values=[1, 3, 5, 10]):
        super().__init__()
        self.test_data = test_data
        self.user_map = user_map
        self.game_map = game_map
        self.game_max_lengths = game_max_lengths
        self.top_k_values = top_k_values
    
    def on_epoch_end(self, epoch, logs=None):
        metrics = compute_ranking_metrics(
            self.model, self.test_data, self.user_map, self.game_map, 
            self.game_max_lengths, self.top_k_values
        )
        
        print(f"\n📊 Epoch {epoch + 1} - Ranking Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            # Log to tensorboard if needed
            if logs is not None:
                logs[f'val_{metric}'] = value

# ------------------------------------------------
# Métricas
# ------------------------------------------------
def precision_at_k(ranked, ground_truth, k):
    ranked_k = ranked[:k]
    return len(set(ranked_k) & set(ground_truth)) / k

def recall_at_k(ranked, ground_truth, k):
    ranked_k = ranked[:k]
    return len(set(ranked_k) & set(ground_truth)) / len(ground_truth)

def ndcg_at_k(ranked, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def tf_auc_score(all_true, all_pred):    
    # Usar AUC do TensorFlow
    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(all_true, all_pred)
    return auc_metric.result().numpy()


def compute_roc_auc(y_true, y_score):
    """Compute ROC curve and AUC manually using numpy."""
    import numpy as _np
    y_true_arr = _np.array(y_true, dtype=_np.float32)
    y_score_arr = _np.array(y_score, dtype=_np.float32)
    if len(y_true_arr) == 0:
        return _np.array([0.0, 1.0]), _np.array([0.0, 1.0]), 0.0
    # sort by score descending
    desc = _np.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[desc]
    P = _np.sum(y_true_sorted)
    N = len(y_true_sorted) - P
    if P == 0 or N == 0:
        fpr = _np.array([0.0, 1.0])
        tpr = _np.array([0.0, 1.0])
        auc_val = 0.0
    else:
        tpr = _np.cumsum(y_true_sorted) / P
        fpr = _np.cumsum(1 - y_true_sorted) / N
        tpr = _np.concatenate(([0.0], tpr))
        fpr = _np.concatenate(([0.0], fpr))
        auc_val = _np.trapz(tpr, fpr)
    return fpr, tpr, auc_val

# ------------------------------------------------
# Callback
# ------------------------------------------------
def compute_ranking_metrics(model, test_user_lists, user_map, game_map, game_max_lengths, top_k_values=[1, 3, 5, 10]):
    """Compute ranking metrics on test set including Hits@K"""
    precisions = {k: [] for k in top_k_values}
    recalls = {k: [] for k in top_k_values}
    ndcgs = {k: [] for k in top_k_values}
    hits = {k: [] for k in top_k_values}
    all_true = []
    all_pred = []
    
    for steamid, appids, labels in test_user_lists:
        if len(appids) == 0:
            continue
            
        # Get user features
        user_data = user_map.get(str(steamid), {})
        user_features = {
            'user_lang_id': np.array([user_data.get('lang_id', 0)] * len(appids)),
            'user_top_genre': np.array([user_data.get('top_genre', 0)] * len(appids)),
            'user_top_cat': np.array([user_data.get('top_cat', 0)] * len(appids)),
            'user_top_tag': np.array([user_data.get('top_tag', 0)] * len(appids)),
            'user_genre_dominance': np.array([[user_data.get('genre_dominance', 0.0)]] * len(appids)),
            'user_genre_diversity': np.array([[user_data.get('genre_diversity', 0.0)]] * len(appids))
        }
        
        # Get game features
        game_features = {
            'game_genre_ids': [],
            'game_category_ids': [],
            'game_user_tag_ids': [],
            'game_review_score': []
        }
        
        for appid in appids:
            game = game_map.get(int(appid), {})
            game_features['game_genre_ids'].append(
                _pad_list(game.get('genre_ids', []), game_max_lengths['genre_ids'])
            )
            game_features['game_category_ids'].append(
                _pad_list(game.get('category_ids', []), game_max_lengths['category_ids'])
            )
            game_features['game_user_tag_ids'].append(
                _pad_list(game.get('user_tag_ids', []), game_max_lengths['user_tag_ids'])
            )
            game_features['game_review_score'].append([game.get('review_score', 0.0)])
        
        # Convert to arrays
        for key in ['game_genre_ids', 'game_category_ids', 'game_user_tag_ids']:
            game_features[key] = np.array(game_features[key])
        game_features['game_review_score'] = np.array(game_features['game_review_score'])
        
        # Prepare batch dimension
        inputs = {**user_features, **game_features}
        for key in inputs:
            inputs[key] = np.expand_dims(inputs[key], 0)  # Add batch dimension
        
        # Get predictions
        predictions = model.predict(inputs, verbose=0)
        scores = predictions['score'][0]  # Remove batch dimension
        
        # Get ground truth positive items
        ground_truth_items = [appids[i] for i, label in enumerate(labels) if label == 1]
        if not ground_truth_items:
            continue
            
        # Rank items by predicted score
        ranked_items = [appids[i] for i in np.argsort(scores)[::-1]]
        
        # Store for AUC
        all_true.extend(labels)
        all_pred.extend(scores)
        
        # Compute metrics for each k
        for k in top_k_values:
            if k <= len(ranked_items):
                precisions[k].append(precision_at_k(ranked_items, ground_truth_items, k))
                recalls[k].append(recall_at_k(ranked_items, ground_truth_items, k))
                ndcgs[k].append(ndcg_at_k(ranked_items, ground_truth_items, k))
                hits_k = 1 if len(set(ranked_items[:k]) & set(ground_truth_items)) > 0 else 0
                hits[k].append(hits_k)
    
    # Compute mean metrics
    mean_metrics = {}
    for k in top_k_values:
        if precisions[k]:
            mean_metrics[f'precision@{k}'] = np.mean(precisions[k])
            mean_metrics[f'recall@{k}'] = np.mean(recalls[k])
            mean_metrics[f'ndcg@{k}'] = np.mean(ndcgs[k])        
        if hits[k]:
            mean_metrics[f'hits@{k}'] = np.mean(hits[k])

    mean_metrics['auc'] = tf_auc_score(
        tf.constant(all_true, dtype=tf.float32),
        tf.constant(all_pred, dtype=tf.float32)
    )
    
    return mean_metrics

# Add the _pad_list function that's used in compute_ranking_metrics
def _pad_list(lst, max_len, pad_value=0):
    if lst is None:
        lst = []
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))

# ------------------------------------------------
# Funções para salvar e carregar pickle
# ------------------------------------------------
def save_pickle(data, file_path):
    """Salva dados em arquivo pickle"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Dados salvos em: {file_path}")

def load_pickle(file_path):
    """Carrega dados de arquivo pickle"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"📂 Dados carregados de: {file_path}")
    return data

def pickle_files_exist():
    """Verifica se todos os arquivos pickle necessários existem"""
    required_files = [
        GAME_MAP_PATH, 
        USER_MAP_PATH, 
        UNIQUE_LANGS_PATH,
        GAME_MAX_LENGTHS_PATH,
        UNIQUE_IDS_PATH
    ]
    return all(os.path.exists(f) for f in required_files)

# ------------------------------------------------
# Plotagem de gráficos de métricas
# ------------------------------------------------

def save_history_txt(history, file_path):
    """Escreve o histórico de treino em um arquivo de texto e adiciona máximos das métricas."""
    with open(file_path, 'w') as f:
        f.write('Epoch')
        for key in history.history.keys():
            f.write(f', {key}')
        f.write('\n')
        num_epochs = len(next(iter(history.history.values())))
        for i in range(num_epochs):
            f.write(str(i + 1))
            for key in history.history.keys():
                f.write(f', {history.history[key][i]:.6f}')
            f.write('\n')
        f.write('\n# Máximos por métrica\n')
        for key, values in history.history.items():
            f.write(f'{key}: {max(values):.6f}\n')
    print(f"📄 Histórico de treino salvo em: {file_path}")
def plot_metric(train_key, val_key, title, ylabel, history, epochs, save_path):
    """Bar chart version of the metric plot."""
    plt.figure(figsize=(10, 4))
    train = history.history.get(train_key, [])
    val = history.history.get(val_key, [])
    # convert epochs to numpy array for offsetting
    epochs_arr = np.array(list(epochs))
    width = 0.35
    if train:
        plt.bar(epochs_arr - width/2, train, width, label='Treino', alpha=0.7)
    if val:
        plt.bar(epochs_arr + width/2, val, width, label='Validação', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    
def plot_ranking_metrics(ranking_metrics, save_path):
    """Plotar métricas de ranking @K como gráficos de barras"""
    plt.figure(figsize=(12, 8))
    
    # Extrair métricas por tipo
    precisions = {k: v for k, v in ranking_metrics.items() if k.startswith('precision@')}
    recalls = {k: v for k, v in ranking_metrics.items() if k.startswith('recall@')}
    ndcgs = {k: v for k, v in ranking_metrics.items() if k.startswith('ndcg@')}
    
    # Extrair valores K
    k_values = [int(k.split('@')[1]) for k in precisions.keys()]
    k_values.sort()
    width = 0.4
    idx = np.arange(len(k_values))
    
    # Plotar Precision@K
    plt.subplot(2, 2, 1)
    precision_values = [precisions[f'precision@{k}'] for k in k_values]
    plt.bar(idx, precision_values, width, color='blue', alpha=0.7)
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.title('Precision@K')
    plt.xticks(idx, k_values)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plotar Recall@K
    plt.subplot(2, 2, 2)
    recall_values = [recalls[f'recall@{k}'] for k in k_values]
    plt.bar(idx, recall_values, width, color='green', alpha=0.7)
    plt.xlabel('K')
    plt.ylabel('Recall')
    plt.title('Recall@K')
    plt.xticks(idx, k_values)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plotar NDCG@K
    plt.subplot(2, 2, 3)
    ndcg_values = [ndcgs[f'ndcg@{k}'] for k in k_values]
    plt.bar(idx, ndcg_values, width, color='red', alpha=0.7)
    plt.xlabel('K')
    plt.ylabel('NDCG')
    plt.title('NDCG@K')
    plt.xticks(idx, k_values)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Adicionar AUC como texto
    plt.subplot(2, 2, 4)
    auc_value = ranking_metrics.get('auc', 0.0)
    plt.text(0.5, 0.5, f'AUC: {auc_value:.4f}', 
            fontsize=16, ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    plt.title('AUC Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------
# main
# ------------------------------------------------
def main():
    print(f"🚀 Iniciando treinamento do modelo - Treino Nº {TRAIN_NUM}")
    print("📦 Carregando base de interações e jogos...")
    
    # Carregar dados usando as novas funções
    interactions_df = load_interactions_dataframe(INTERACTIONS_FILE)
    games_df = load_games_dataframe(GAMES_CSV)
    
    # Construir mapeamentos
    # Verificar se os pickle files existem
    if pickle_files_exist():
        print("🔄 Carregando mapeamentos de arquivos pickle...")
        game_map = load_pickle(GAME_MAP_PATH)
        user_map = load_pickle(USER_MAP_PATH)
        unique_langs = load_pickle(UNIQUE_LANGS_PATH)
        game_max_lengths = load_pickle(GAME_MAX_LENGTHS_PATH)
        unique_ids = load_pickle(UNIQUE_IDS_PATH)
        
        unique_game_gens = unique_ids['unique_game_gens']
        unique_game_cats = unique_ids['unique_game_cats']
        unique_game_tags = unique_ids['unique_game_tags']
        
    else:
        print("🗺️ Construindo mapeamento de jogos...")
        game_map = build_game_map(games_df)
        save_pickle(game_map, GAME_MAP_PATH)
        
        print("🗺️ Construindo mapeamento de usuários...")
        user_map, unique_langs = build_user_map(interactions_df, game_map, top_n=TOP_N)
        save_pickle(user_map, USER_MAP_PATH)
        save_pickle(unique_langs, UNIQUE_LANGS_PATH)
        
        # Calcular game_max_lengths
        game_max_lengths = {
            'genre_ids': max(len(game['genre_ids']) for game in game_map.values()) if game_map else 1,
            'category_ids': max(len(game['category_ids']) for game in game_map.values()) if game_map else 1,
            'user_tag_ids': max(len(game['user_tag_ids']) for game in game_map.values()) if game_map else 1
        }
        save_pickle(game_max_lengths, GAME_MAX_LENGTHS_PATH)
        
        # Calcular valores únicos para embeddings
        unique_game_gens = max([max(game['genre_ids']) for game in game_map.values() if game['genre_ids']], default=0) + 1
        unique_game_cats = max([max(game['category_ids']) for game in game_map.values() if game['category_ids']], default=0) + 1
        unique_game_tags = max([max(game['user_tag_ids']) for game in game_map.values() if game['user_tag_ids']], default=0) + 1
        
        unique_ids = {
            'unique_game_gens': unique_game_gens,
            'unique_game_cats': unique_game_cats,
            'unique_game_tags': unique_game_tags
        }
        save_pickle(unique_ids, UNIQUE_IDS_PATH)
    
    # Calcular game_max_lengths
    game_max_lengths = {
        'genre_ids': max(len(game['genre_ids']) for game in game_map.values()) if game_map else 1,
        'category_ids': max(len(game['category_ids']) for game in game_map.values()) if game_map else 1,
        'user_tag_ids': max(len(game['user_tag_ids']) for game in game_map.values()) if game_map else 1
    }
    
    # Calcular valores únicos para embeddings
    unique_game_gens = max([max(game['genre_ids']) for game in game_map.values() if game['genre_ids']], default=0) + 1
    unique_game_cats = max([max(game['category_ids']) for game in game_map.values() if game['category_ids']], default=0) + 1
    unique_game_tags = max([max(game['user_tag_ids']) for game in game_map.values() if game['user_tag_ids']], default=0) + 1

    # ---------------- Split por usuário ----------------
    print("🔀 Dividindo dados de treino e teste por usuário...")
    unique_users = list(user_map.keys())
    np.random.seed(42)
    np.random.shuffle(unique_users)
    n_test_users = max(1, int(len(unique_users) * VAL_SPLIT))
    test_users = set(unique_users[:n_test_users])
    train_users = set(unique_users[n_test_users:])

    # Filtrar interações por usuários de treino/teste
    train_interactions = interactions_df[interactions_df['steamid'].astype(str).isin(train_users)]
    test_interactions = interactions_df[interactions_df['steamid'].astype(str).isin(test_users)]

    # ---------------- Construir listas de usuários ----------------
    print("📝 Construindo listas de usuários...")
    train_user_lists = build_user_lists(train_interactions, game_map, neg_k=NEG_K, top_n=TOP_N, seed=42)
    test_user_lists = build_user_lists(test_interactions, game_map, neg_k=NEG_K, top_n=TOP_N, seed=42)
    debug_labels(train_user_lists, "Train")
    debug_labels(test_user_lists, "Test")
    debug_negative_sampling(train_user_lists)
    debug_negative_sampling(test_user_lists)

    # ---------------- Criar datasets TensorFlow ----------------
    print("⚙️  Preparando datasets TensorFlow...")
    ds_train = create_tf_dataset(
        train_user_lists, user_map, game_map, game_max_lengths, 
        batch_size=BATCH_SIZE, shuffle=True, top_n=TOP_N
    )
    ds_test = create_tf_dataset(
        test_user_lists, user_map, game_map, game_max_lengths, 
        batch_size=BATCH_SIZE, shuffle=False, top_n=TOP_N
    )

    # ---------------- Criar modelo ----------------
    print("🧠 Criando modelo Two-Tower...")
    # user_max_lengths não é mais usado no modelo, mas mantemos por compatibilidade
    user_max_lengths = {}  # Dicionário vazio já que não é mais usado
    
    model = create_two_tower_model(
        user_max_lengths=user_max_lengths,
        game_max_lengths=game_max_lengths,
        embedding_dim=EMBEDDING_DIM,
        unique_game_gens=unique_game_gens,
        unique_game_cats=unique_game_cats,
        unique_game_tags=unique_game_tags,
        unique_langs=unique_langs,
        top_n=TOP_N
    )

    # ---------------- Compilar ----------------
    print("⚗️  Compilando modelo...")
    map_metric = tfr.keras.metrics.MeanAveragePrecisionMetric(name="map_metric", topn=TOP_N + NEG_K)
    ndcg_metric = tfr.keras.metrics.NDCGMetric(name="ndcg_metric", topn=TOP_N + NEG_K)

    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-3)
    losses = {'score': tf.keras.losses.BinaryCrossentropy(from_logits=True)}
    loss_weights = {'score': 1.0}
    metrics = {'score': [map_metric, ndcg_metric]}

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    # ---------------- Callbacks ----------------
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=8, 
            min_delta=1e-4, 
            mode='min',
            restore_best_weights=True, 
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    #DEBUG START
    print("\n" + "="*50)
    print("🔍 DEBUG")
    print("="*50)
    
    # 1. Verificar dados básicos
    print(f"📊 Interações totais: {len(interactions_df)}")
    print(f"📊 Jogos no game_map: {len(game_map)}")
    print(f"📊 Usuários no user_map: {len(user_map)}")
    
    # 2. Verificar divisão train/test
    print(f"📊 Train interactions: {len(train_interactions)}")
    print(f"📊 Test interactions: {len(test_interactions)}")
    print(f"📊 Train user lists: {len(train_user_lists)}")
    print(f"📊 Test user lists: {len(test_user_lists)}")
    
    # 3. Verificar leakage
    check_data_leakage(train_interactions, test_interactions, user_map, game_map)
    
    # 4. Verificar labels
    debug_labels(train_user_lists, "Train")
    debug_labels(test_user_lists, "Test")
    
    # 5. Sanity check do modelo
    simple_sanity_check(model, user_map, game_map, game_max_lengths)
    
    print("="*50)
    #DEBUG END

    # ---------------- Treinar ----------------
    print("🚀 Iniciando treinamento...")
    history = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    print("✅ Treinamento concluído.")
    model.save(MODEL_SAVE_PATH)

    # salvar histórico em texto
    save_history_txt(history, HISTORY_LOG)

    # ---------------- Avaliar ----------------
    print("📊 Avaliando modelo no conjunto de teste...")
    results = model.evaluate(ds_test, verbose=1)
    print(f"Resultados de avaliação: {results}")

    # ---------------- Curva ROC ----------------
    print("📈 Gerando curva ROC...")
    y_true = []
    y_pred = []
    # percorrer dataset de teste uma vez
    for batch in ds_test:
        inputs, labels = batch
        preds = model.predict(inputs, verbose=0)
        # handle both tensor and numpy outputs
        scores = preds['score']
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        y_pred.extend(scores.flatten().tolist())
        lbls = labels['score']
        if hasattr(lbls, 'numpy'):
            lbls = lbls.numpy()
        y_true.extend(lbls.flatten().tolist())
    if y_true and y_pred:
        fpr, tpr, roc_auc = compute_roc_auc(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(ROC_PLOT)
        plt.close()
        print(f"📂 Curva ROC salva em: {ROC_PLOT}")
    else:
        print("⚠️ Não foi possível calcular ROC (sem dados de verdadeiros ou previsões)")

    print("📊 Computando métricas de ranking...")
    ranking_metrics = compute_ranking_metrics(
        model, test_user_lists, user_map, game_map, game_max_lengths, 
        top_k_values=[5, 10, 20]
    )

    print("🎯 Métricas de Ranking:")
    for metric, value in ranking_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Also update the plotting section to include AUC if computed
    if 'auc' in ranking_metrics:
        print(f"  AUC: {ranking_metrics['auc']:.4f}")

    # ---------------- Gráficos ----------------
    print("📈 Gerando gráficos de métricas...")
    epochs = range(1, len(history.history.get('loss', [])) + 1)

    plot_metric('loss', 'val_loss', 'Loss x Epoch', 'Loss', history, epochs, LOSS_PLOT)
    plot_metric('map_metric', 'val_map_metric', 'MAP x Epoch', 'MAP', history, epochs, MAP_PLOT)
    plot_metric('ndcg_metric', 'val_ndcg_metric', 'NDCG x Epoch', 'NDCG', history, epochs, NDCG_PLOT)
    plot_ranking_metrics(ranking_metrics, RANKING_METRICS_PLOT)

    print(f"📂 Gráficos salvos em:\n{LOSS_PLOT}\n{MAP_PLOT}\n{NDCG_PLOT}\n{RANKING_METRICS_PLOT}\n")

if __name__ == '__main__':
    main()