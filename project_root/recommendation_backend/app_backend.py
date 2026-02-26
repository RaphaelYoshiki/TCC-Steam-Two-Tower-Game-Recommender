from fastapi import FastAPI
from pydantic import BaseModel

from recommendation_backend.model_loader import load_artifacts
from recommendation_backend.recommend import recommend_games
from recommendation_backend.game_lookup import GameLookup

app = FastAPI()

model, game_map, game_max_lengths = load_artifacts()
lookup = GameLookup()

class UserProfile(BaseModel):
    lang_id: int
    top_genre: int
    top_cat: int
    top_tag: int
    genre_dominance: float
    genre_diversity: float

@app.post("/recommend")
def recommend(profile: UserProfile):
    print("🔥 PROFILE RECEBIDO:", profile)
    appids = recommend_games(
        model=model,
        user_profile=profile.dict(),
        game_map=game_map,
        game_max_lengths=game_max_lengths,
        top_k=10
    )

    games = lookup.resolve(appids)

    return {
        "recommendations": games
    }
