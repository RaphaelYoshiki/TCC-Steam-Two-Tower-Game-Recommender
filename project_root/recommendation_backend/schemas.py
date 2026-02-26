from pydantic import BaseModel

class UserProfile(BaseModel):
    lang_id: int
    top_genre: int
    top_cat: int
    top_tag: int
    genre_dominance: float
    genre_diversity: float

class RecommendationResponse(BaseModel):
    appid: int
    name: str
    score: float
