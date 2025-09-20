# utils/game_recommender.py
import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = "utils/all_data/gr_two_tower.keras"
GAME_DF_PATH = "utils/all_data/treated_dataframe.csv"

def pad_lists(df, list_columns):
    df_copy = df.copy()
    max_lengths = {}
    for col in list_columns:
        max_lengths[col] = max(len(lst) for lst in df_copy[col])
    for col, max_len in max_lengths.items():
        df_copy[col] = df_copy[col].apply(
            lambda lst: lst + [0] * (max_len - len(lst)) if len(lst) < max_len else lst[:max_len]
        )
    return df_copy, max_lengths

def _load_games():
    game_df = pd.read_csv(GAME_DF_PATH)
    for col in ["developer_ids", "category_ids", "genre_ids", "user_tag_ids"]:
        game_df[col] = game_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    game_df_padded, game_max_lengths = pad_lists(game_df, ["developer_ids", "category_ids", "genre_ids", "user_tag_ids"])
    return game_df, game_df_padded, game_max_lengths

# Carregar modelo treinado
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
game_df, game_df_padded, game_max_lengths = _load_games()

def _prepare_game_inputs():
    return {
        "game_developer_ids": np.expand_dims(np.array(game_df_padded["developer_ids"].tolist(), dtype=np.int32), axis=1),
        "game_category_ids": np.expand_dims(np.array(game_df_padded["category_ids"].tolist(), dtype=np.int32), axis=1),
        "game_genre_ids": np.expand_dims(np.array(game_df_padded["genre_ids"].tolist(), dtype=np.int32), axis=1),
        "game_user_tag_ids": np.expand_dims(np.array(game_df_padded["user_tag_ids"].tolist(), dtype=np.int32), axis=1),
        "game_review_score": np.expand_dims(np.array(game_df_padded["review_score"].tolist(), dtype=np.float32).reshape(-1, 1), axis=1),
    }

GAME_INPUTS_ALL = _prepare_game_inputs()

def _pad_profile_list(lst, max_len):
    lst = lst or []
    if len(lst) < max_len:
        return lst + [0]*(max_len - len(lst))
    return lst[:max_len]

def recommend_from_profile(profile: dict, top_k: int = 10):
    """
    Gera recomendações a partir de um perfil anônimo.
    """
    n_games = len(game_df_padded)

    # pad user lists to same max_lengths used in treino
    u_gen = _pad_profile_list(profile.get("interacted_genres", []), max_len=game_max_lengths["genre_ids"])
    u_cat = _pad_profile_list(profile.get("interacted_categories", []), max_len=game_max_lengths["category_ids"])
    u_tag = _pad_profile_list(profile.get("interacted_tags", []), max_len=game_max_lengths["user_tag_ids"])
    u_dev = _pad_profile_list(profile.get("interacted_developers", []), max_len=game_max_lengths["developer_ids"])

    # cada jogo recebe o mesmo perfil de usuário mas precisa ter shape (n_games,1,seq_len)
    user_inputs = {
        "user_interacted_genres": np.repeat(np.expand_dims([u_gen], axis=1), n_games, axis=0),
        "user_interacted_categories": np.repeat(np.expand_dims([u_cat], axis=1), n_games, axis=0),
        "user_interacted_tags": np.repeat(np.expand_dims([u_tag], axis=1), n_games, axis=0),
        "user_interacted_developers": np.repeat(np.expand_dims([u_dev], axis=1), n_games, axis=0),
    }

    game_inputs = GAME_INPUTS_ALL

    # unir inputs user+game no dicionário que seu modelo espera
    model_inputs = {
        **user_inputs,
        **game_inputs
    }

    preds = model.predict(model_inputs, verbose=0)
    preds_flat = np.array(preds).reshape(-1)

    top_idx = np.argsort(-preds_flat)[:top_k]
    out = game_df.iloc[top_idx].copy().reset_index(drop=True)
    out["score"] = preds_flat[top_idx]
    return out[["appid", "name", "score"]]

__all__ = ["recommend_from_profile", "game_df"]
