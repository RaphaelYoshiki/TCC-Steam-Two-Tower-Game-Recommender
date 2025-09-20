# data_prep.py
import ast, numpy as np, pandas as pd, tensorflow as tf
from typing import Tuple, Dict, Any
import os, multiprocessing

def _safe_literal_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []

def load_games_dataframe(games_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(games_csv_path, encoding='utf-8')
    list_cols = ['developer_ids', 'category_ids', 'genre_ids', 'user_tag_ids']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_literal_eval).apply(lambda v: list(v) if v is not None else [])
        else:
            df[col] = [[] for _ in range(len(df))]
    if 'review_score' not in df.columns:
        df['review_score'] = 0.0
    return df

def load_interactions_dataframe(interactions_csv_path: str):
    df = pd.read_csv(interactions_csv_path, encoding='utf-8')
    if 'appid' not in df.columns or 'steamid' not in df.columns:
        raise ValueError("interactions CSV must contain 'steamid' and 'appid' columns")
    return df

def build_game_map(game_df: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
    game_map = {}
    for _, row in game_df.iterrows():
        appid = row['game_id']
        game_map[appid] = {
            'developer_ids': row.get('developer_ids', []) or [],
            'category_ids': row.get('category_ids', []) or [],
            'genre_ids': row.get('genre_ids', []) or [],
            'user_tag_ids': row.get('user_tag_ids', []) or [],
            'review_score': float(row.get('review_score', 0.0) or 0.0)
        }
    return game_map

def _pad_list(lst, max_len, pad_value=0):
    if lst is None:
        lst = []
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))

def summarize_max_values(game_df: pd.DataFrame):
    maxes = {}
    for col in ['genre_ids', 'category_ids', 'user_tag_ids', 'developer_ids']:
        all_vals = []
        if col in game_df.columns:
            for lst in game_df[col]:
                if lst: all_vals.extend([int(x) for x in lst])
        maxes[col] = max(all_vals) if len(all_vals) > 0 else 0
    return maxes

def build_user_lists(interactions_df: pd.DataFrame, game_map: dict, neg_k: int = 5):
    """
    Para cada usuário, retorna listas [positivo, neg1, neg2, ...].
    """
    all_appids = list(game_map.keys())
    grouped = interactions_df.groupby("steamid")
    user_lists = []

    for steamid, group in grouped:
        positives = list(group["appid"].unique())
        for pos_app in positives:
            # gerar k negativos (garantir que não estejam nos positivos)
            neg_pool = [a for a in all_appids if a not in positives]
            if len(neg_pool) < neg_k:
                continue
            neg_samples = np.random.choice(neg_pool, size=neg_k, replace=False)

            candidates = [pos_app] + list(neg_samples)
            labels = [1.0] + [0.0] * neg_k
            user_lists.append((steamid, candidates, labels))
    return user_lists

def create_tf_dataset(user_lists, game_map, user_max_lengths, game_max_lengths,
                      batch_size=128, shuffle=True):
    """
    Cria tf.data.Dataset no formato:
      inputs: dict de tensores com shape (batch, list_size, seq_len ou 1)
      labels: (batch, list_size)
    """
    user_in_cols = ['interacted_genres', 'interacted_categories', 'interacted_tags', 'interacted_developers']
    game_in_cols = ['genre_ids', 'category_ids', 'user_tag_ids', 'developer_ids']

    list_size = len(user_lists[0][1])
    n = len(user_lists)

    # Pré-aloca arrays
    user_arrays = {col: np.zeros((n, list_size, user_max_lengths[col]), dtype=np.int32) for col in user_in_cols}
    game_arrays = {col: np.zeros((n, list_size, game_max_lengths[col]), dtype=np.int32) for col in game_in_cols}
    review_scores = np.zeros((n, list_size, 1), dtype=np.float32)
    labels = np.zeros((n, list_size), dtype=np.float32)

    for i, (steamid, appids, lbls) in enumerate(user_lists):
        # features do usuário = histórico prévio (simples: todos os positivos menos o atual)
        user_feats = {c: [] for c in user_in_cols}
        for past in appids:  # aqui poderia ser refinado para histórico temporal
            g = game_map.get(past, {})
            user_feats['interacted_genres'].extend(g.get('genre_ids', []))
            user_feats['interacted_categories'].extend(g.get('category_ids', []))
            user_feats['interacted_tags'].extend(g.get('user_tag_ids', []))
            user_feats['interacted_developers'].extend(g.get('developer_ids', []))

        for j, appid in enumerate(appids):
            gf = game_map.get(appid, {'developer_ids': [], 'category_ids': [], 'genre_ids': [], 'user_tag_ids': [], 'review_score': 0.0})
            for col in user_in_cols:
                arr = _pad_list(user_feats.get(col, []), user_max_lengths[col], 0)
                user_arrays[col][i, j] = np.array(arr, dtype=np.int32)
            for col in game_in_cols:
                arr = _pad_list(gf.get(col, []), game_max_lengths[col], 0)
                game_arrays[col][i, j] = np.array(arr, dtype=np.int32)
            review_scores[i, j, 0] = float(gf.get('review_score', 0.0))
            labels[i, j] = lbls[j]

    inputs = {
        'user_interacted_genres': user_arrays['interacted_genres'],
        'user_interacted_categories': user_arrays['interacted_categories'],
        'user_interacted_tags': user_arrays['interacted_tags'],
        'user_interacted_developers': user_arrays['interacted_developers'],
        'game_genre_ids': game_arrays['genre_ids'],
        'game_category_ids': game_arrays['category_ids'],
        'game_user_tag_ids': game_arrays['user_tag_ids'],
        'game_developer_ids': game_arrays['developer_ids'],
        'game_review_score': review_scores
    }
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, n))
    ds = ds.batch(batch_size).prefetch(1)  # fixo para evitar 100% CPU
    return ds

def load_and_preprocess_interactions(interactions_csv_path: str, games_csv_path: str,
                                     batch_size: int = 128, neg_k: int = 5):
    game_df = load_games_dataframe(games_csv_path)
    interactions_df = load_interactions_dataframe(interactions_csv_path)
    game_map = build_game_map(game_df)

    user_lists = build_user_lists(interactions_df, game_map, neg_k=neg_k)
    if len(user_lists) == 0:
        raise ValueError("Nenhuma interação válida encontrada.")

    # calcular max_lengths
    user_list_cols = ['interacted_genres', 'interacted_categories', 'interacted_tags', 'interacted_developers']
    game_list_cols = ['genre_ids', 'category_ids', 'user_tag_ids', 'developer_ids']
    user_max_lengths = {col: 16 for col in user_list_cols}   # truncamento fixo
    game_max_lengths = {col: 16 for col in game_list_cols}

    max_vals = summarize_max_values(game_df)
    max_values = {
        'unique_game_gens': max_vals['genre_ids'] + 1,
        'unique_game_cats': max_vals['category_ids'] + 1,
        'unique_game_tags': max_vals['user_tag_ids'] + 1,
        'unique_game_devs': max_vals['developer_ids'] + 1
    }

    ds = create_tf_dataset(user_lists, game_map, user_max_lengths, game_max_lengths,
                           batch_size=batch_size, shuffle=True)
    return ds, user_max_lengths, game_max_lengths, max_values
