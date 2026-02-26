import ast, numpy as np, pandas as pd, tensorflow as tf, json
from collections import Counter

# ------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------
def _safe_literal_eval(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []

def _pad_list(lst, max_len, pad_value=0):
    if lst is None:
        lst = []
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))

# ------------------------------------------------------------
# Carregamento de dados
# ------------------------------------------------------------
def load_games_dataframe(games_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(games_csv_path, encoding='utf-8')
    list_cols = ['category_ids', 'genre_ids', 'user_tag_ids']
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
        raise ValueError("interactions CSV must contain 'steamid' and 'appid'")
    return df

# ------------------------------------------------------------
# Mapeamento de jogos e usuários
# ------------------------------------------------------------
def build_game_map(game_df: pd.DataFrame):
    game_map = {}
    for _, row in game_df.iterrows():
        appid = int(row['game_id'])
        game_map[appid] = {
            'category_ids': row.get('category_ids', []) or [],
            'genre_ids': row.get('genre_ids', []) or [],
            'user_tag_ids': row.get('user_tag_ids', []) or [],
            'review_score': float(row.get('review_score', 0.0) or 0.0)
        }
    return game_map

def build_user_map(interactions_df: pd.DataFrame, game_map: dict, top_n: int = 10):
    grouped = interactions_df.groupby('steamid')
    users = []
    langs_map = {}

    for steamid, group in grouped:
        # Contadores para gêneros, categorias e tags
        genre_counter = Counter()
        cat_counter = Counter()
        tag_counter = Counter()
        
        for _, row in group.iterrows():
            appid = int(row['appid'])
            game = game_map.get(appid, {})
            
            # Contar ocorrências
            for genre_id in game.get('genre_ids', []):
                genre_counter[genre_id] += 1
            for cat_id in game.get('category_ids', []):
                cat_counter[cat_id] += 1
            for tag_id in game.get('user_tag_ids', []):
                tag_counter[tag_id] += 1
        
        # Obter os mais frequentes
        top_genre = genre_counter.most_common(1)[0][0] if genre_counter else 0
        top_cat = cat_counter.most_common(1)[0][0] if cat_counter else 0
        top_tag = tag_counter.most_common(1)[0][0] if tag_counter else 0
        
        # Calcular dominância e diversidade
        total_genre_count = sum(genre_counter.values())
        genre_dominance = genre_counter[top_genre] / total_genre_count if total_genre_count > 0 else 0
        
        # Linguagem
        lang = group['language'].mode()[0] if not group['language'].isna().all() else 'unknown'
        if lang not in langs_map:
            langs_map[lang] = len(langs_map)

        users.append({
            'steamid': str(steamid),
            'lang_id': langs_map[lang],
            'top_genre': top_genre,
            'top_cat': top_cat, 
            'top_tag': top_tag,
            'genre_dominance': genre_dominance,
            'genre_diversity': len(genre_counter)
        })        
    
    with open("./recommendation/aux_files/lang_id_map.json", "w", encoding="utf-8") as f:
        json.dump(langs_map, f, ensure_ascii=False, indent=4)

    print("Language ID map salvo em recommendation/aux_files/lang_id_map.json")

    user_map = {u['steamid']: u for u in users}
    return user_map, len(langs_map) + 1

# ------------------------------------------------------------
# Negative sampling e dataset TF - UPDATED VERSION
# ------------------------------------------------------------
def build_user_lists(interactions_df: pd.DataFrame, game_map: dict, neg_k: int = 10, top_n: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    grouped = interactions_df.groupby("steamid")
    user_lists = []
    
    # Coletar jogos populares para amostragem
    popular_games = set(interactions_df['appid'].value_counts().head(500).index)
    
    for steamid, group in grouped:
        all_pos_appids = group[group["voted_up"] == True]["appid"].astype(int).tolist()
        
        if len(all_pos_appids) == 0:
            continue

        n_pos = min(top_n, len(all_pos_appids))
        pos_appids = rng.choice(all_pos_appids, size=n_pos, replace=False)
        user_played = set(group['appid'].astype(int))
        
        # Estratégia mista de negativos
        neg_candidates = []
        
        # 1. Negativos "hard": jogos populares que o usuário não jogou
        hard_negs = popular_games - user_played
        if len(hard_negs) > 0:
            neg_candidates.extend(list(hard_negs))
        
        # 2. Negativos aleatórios (fallback)
        all_games = set(game_map.keys())
        random_negs = all_games - user_played
        if len(neg_candidates) < neg_k:
            needed = neg_k - len(neg_candidates)
            additional = rng.choice(list(random_negs), 
                                  size=min(needed, len(random_negs)), 
                                  replace=False)
            neg_candidates.extend(additional)
        
        n_neg = min(neg_k, len(neg_candidates))
        neg_appids = rng.choice(neg_candidates, size=n_neg, replace=False)
        
        appid_list = np.concatenate([pos_appids, neg_appids])
        label_list = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        
        user_lists.append((steamid, appid_list, label_list))

    return user_lists

def create_tf_dataset(user_lists, user_map, game_map, game_max_lengths, batch_size=128, shuffle=True, top_n=5):
    # Ensure we have a consistent list_size across all examples
    if not user_lists:
        list_size = 1
    else:
        list_size = max(len(appids) for _, appids, _ in user_lists)
    
    print(f"📏 Using fixed list_size: {list_size}")

    def gen():
        for steamid, appids, lbls in user_lists:
            u = user_map.get(str(steamid), {})
            
            # Pad or truncate to the fixed list_size
            current_size = len(appids)
            if current_size < list_size:
                # Pad with dummy values
                padding_needed = list_size - current_size
                appids = np.concatenate([appids, np.zeros(padding_needed, dtype=np.int32)])
                lbls = np.concatenate([lbls, np.zeros(padding_needed, dtype=np.float32)])
            elif current_size > list_size:
                # Truncate if necessary (shouldn't happen with our max calculation)
                appids = appids[:list_size]
                lbls = lbls[:list_size]
            
            user_inputs = {
                'user_lang_id': np.full((list_size,), u.get('lang_id', 0), dtype=np.int32),
                'user_top_genre': np.full((list_size,), u.get('top_genre', 0), dtype=np.int32),
                'user_top_cat': np.full((list_size,), u.get('top_cat', 0), dtype=np.int32),
                'user_top_tag': np.full((list_size,), u.get('top_tag', 0), dtype=np.int32),
                'user_genre_dominance': np.full((list_size, 1), u.get('genre_dominance', 0.0), dtype=np.float32),
                'user_genre_diversity': np.full((list_size, 1), u.get('genre_diversity', 0.0), dtype=np.float32)
            }

            game_arrays = {'genre_ids': [], 'category_ids': [], 'user_tag_ids': []}
            review_scores = []

            for appid in appids:
                if appid == 0:  # Padding item
                    # Create dummy game features for padding
                    for col in game_arrays.keys():
                        game_arrays[col].append(_pad_list([], game_max_lengths[col]))
                    review_scores.append([0.0])
                else:
                    g = game_map.get(int(appid), {'genre_ids': [], 'category_ids': [], 'user_tag_ids': [], 'review_score': 0.0})
                    for col in game_arrays.keys():
                        game_arrays[col].append(_pad_list(g.get(col, []), game_max_lengths[col]))
                    review_scores.append([float(g.get('review_score', 0.0))])

            inputs = {
                **user_inputs,
                'game_genre_ids': np.array(game_arrays['genre_ids'], dtype=np.int32),
                'game_category_ids': np.array(game_arrays['category_ids'], dtype=np.int32),
                'game_user_tag_ids': np.array(game_arrays['user_tag_ids'], dtype=np.int32),
                'game_review_score': np.array(review_scores, dtype=np.float32)
            }
            labels = {
                'score': np.array(lbls, dtype=np.float32)
            }
            yield inputs, labels

    # Create output signature with fixed shapes
    output_signature = (
        {
            'user_lang_id': tf.TensorSpec(shape=(list_size,), dtype=tf.int32),
            'user_top_genre': tf.TensorSpec(shape=(list_size,), dtype=tf.int32),
            'user_top_cat': tf.TensorSpec(shape=(list_size,), dtype=tf.int32),
            'user_top_tag': tf.TensorSpec(shape=(list_size,), dtype=tf.int32),
            'user_genre_dominance': tf.TensorSpec(shape=(list_size, 1), dtype=tf.float32),
            'user_genre_diversity': tf.TensorSpec(shape=(list_size, 1), dtype=tf.float32),
            'game_genre_ids': tf.TensorSpec(shape=(list_size, game_max_lengths['genre_ids']), dtype=tf.int32),
            'game_category_ids': tf.TensorSpec(shape=(list_size, game_max_lengths['category_ids']), dtype=tf.int32),
            'game_user_tag_ids': tf.TensorSpec(shape=(list_size, game_max_lengths['user_tag_ids']), dtype=tf.int32),
            'game_review_score': tf.TensorSpec(shape=(list_size, 1), dtype=tf.float32)
        },
        {'score': tf.TensorSpec(shape=(list_size,), dtype=tf.float32)}
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(512)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds