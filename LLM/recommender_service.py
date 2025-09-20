# recommender_service.py sem login
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from utils.game_recommender import recommend_from_profile

app = FastAPI(title="Recommender Two-Tower Service")

class RecommendProfile(BaseModel):
    language: Optional[str] = "brazilian"
    interacted_genres: List[int] = []
    interacted_categories: List[int] = []
    interacted_tags: List[int] = []
    interacted_developers: List[int] = []
    top_k: int = 10

@app.post("/recommend")
def recommend(profile: RecommendProfile):
    prof_dict = {
        "language": profile.language,
        "interacted_genres": profile.interacted_genres,
        "interacted_categories": profile.interacted_categories,
        "interacted_tags": profile.interacted_tags,
        "interacted_developers": profile.interacted_developers,
    }
    try:
        recs_df = recommend_from_profile(prof_dict, top_k=profile.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"language": profile.language, "recommendations": recs_df.to_dict(orient="records")}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
