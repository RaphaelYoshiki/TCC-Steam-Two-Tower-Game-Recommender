import numpy as np

def pad_list(lst, max_len):
    return lst[:max_len] + [0] * max(0, max_len - len(lst))

def recommend_games(
    model,
    user_profile,
    game_map,
    game_max_lengths,
    top_k=10
):
    appids = list(game_map.keys())

    # ---------- USER ----------
    list_size = len(appids)
    user_inputs = {
        "user_lang_id": np.full((list_size,), user_profile["lang_id"]),
        "user_top_genre": np.full((list_size,), user_profile["top_genre"]),
        "user_top_cat": np.full((list_size,), user_profile["top_cat"]),
        "user_top_tag": np.full((list_size,), user_profile["top_tag"]),
        "user_genre_dominance": np.full((list_size, 1), user_profile["genre_dominance"]),
        "user_genre_diversity": np.full((list_size, 1), user_profile["genre_diversity"]),
    }

    # ---------- GAMES ----------
    game_inputs = {
        "game_genre_ids": [],
        "game_category_ids": [],
        "game_user_tag_ids": [],
        "game_review_score": []
    }

    for appid in appids:
        g = game_map[appid]
        game_inputs["game_genre_ids"].append(
            pad_list(g["genre_ids"], game_max_lengths["genre_ids"])
        )
        game_inputs["game_category_ids"].append(
            pad_list(g["category_ids"], game_max_lengths["category_ids"])
        )
        game_inputs["game_user_tag_ids"].append(
            pad_list(g["user_tag_ids"], game_max_lengths["user_tag_ids"])
        )
        game_inputs["game_review_score"].append([g["review_score"]])

    inputs = {**user_inputs, **game_inputs}
    for k in inputs:
        inputs[k] = np.expand_dims(np.array(inputs[k]), 0)

    scores = model.predict(inputs, verbose=0)["score"][0]

    ranked = np.argsort(scores)[::-1][:top_k]
    
    print("===== TOP K DEBUG =====")
    for i in ranked:
        print(f"appid={appids[i]} | score={float(scores[i]):.6f}")
    print("=======================")
    
    return [appids[i] for i in ranked]
