import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = "D:/Users/raphakun1010/Documents/Raphael/UFF/TCC/TCC - Sistema de Recomendação/LLM/utils/all_data/gr_two_tower.keras"
GAME_DF_PATH = "D:/Users/raphakun1010/Documents/Raphael/UFF/TCC/TCC - Sistema de Recomendação/LLM/utils/all_data/treated_dataframe.csv"

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

# Carregar modelo
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
game_df, game_df_padded, game_max_lengths = _load_games()

def _prepare_game_inputs():
    return {
        "developer_ids": np.array(game_df_padded["developer_ids"].tolist(), dtype=np.int32),
        "category_ids": np.array(game_df_padded["category_ids"].tolist(), dtype=np.int32),
        "genre_ids": np.array(game_df_padded["genre_ids"].tolist(), dtype=np.int32),
        "user_tag_ids": np.array(game_df_padded["user_tag_ids"].tolist(), dtype=np.int32),
        "review_score": np.array(game_df_padded["review_score"].tolist(), dtype=np.float32).reshape(-1, 1),
    }

GAME_INPUTS_ALL = _prepare_game_inputs()

def recommend_from_profile(profile: dict, top_k: int = 10):
    """
    Gera recomendações a partir de um perfil anônimo.
    profile: {
      "language": "english",
      "interacted_genres": [..],
      "interacted_categories": [..],
      "interacted_tags": [..],
      "interacted_developers": [..]
    }
    """
    n_games = len(game_df_padded)

    # monta inputs do usuário para cada jogo
    user_inputs = {
        "interacted_genres": np.repeat(np.array([profile.get("interacted_genres", [])]), n_games, axis=0),
        "interacted_categories": np.repeat(np.array([profile.get("interacted_categories", [])]), n_games, axis=0),
        "interacted_tags": np.repeat(np.array([profile.get("interacted_tags", [])]), n_games, axis=0),
        "interacted_developers": np.repeat(np.array([profile.get("interacted_developers", [])]), n_games, axis=0),
    }

    game_inputs = GAME_INPUTS_ALL
    preds = model.predict({"user": user_inputs, "game": game_inputs}, verbose=0)
    preds_flat = np.array(preds).reshape(-1)

    top_idx = np.argsort(-preds_flat)[:top_k]
    out = game_df.iloc[top_idx].copy().reset_index(drop=True)
    out["score"] = preds_flat[top_idx]
    return out[["appid", "name", "score"]]

__all__ = ["recommend_from_profile", "game_df"]

profile = {
    "language": "english",
    "interacted_genres": [1, 25, 23, 3, 70],
    "interacted_categories": [2, 28, 23, 62],
    "interacted_tags": [93, 1, 134, 191, 110, 139, 178, 241, 100, 372, 138, 365, 60, 198, 118, 130, 34, 85, 131, 200],
    "interacted_developers": [3851]
}

recs_df = recommend_from_profile(profile, top_k=10)

print(recs_df)