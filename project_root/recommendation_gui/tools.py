RECOMMEND_TOOL = {
    "type": "function",
    "function": {
        "name": "recommend_games",
        "description": "Generate game recommendations based on a structured user profile",
        "parameters": {
            "type": "object",
            "properties": {
                "lang_id": {"type": "integer"},
                "top_genre": {"type": "integer"},
                "top_cat": {"type": "integer"},
                "top_tag": {"type": "integer"},
                "genre_dominance": {"type": "number"},
                "genre_diversity": {"type": "integer"}
            },
            "required": [
                "lang_id",
                "top_genre",
                "top_cat",
                "top_tag",
                "genre_dominance",
                "genre_diversity"
            ]
        }
    }
}
