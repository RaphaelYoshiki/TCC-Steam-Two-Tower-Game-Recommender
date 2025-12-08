# backend.py
import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

import tensorflow as tf

# --------- CONFIG (ajuste caminhos conforme seu ambiente) ----------
MODEL_PATH = os.environ.get("TWO_TOWER_MODEL_PATH", "./models/two_tower_realusers_25.keras")
GAME_MAP_PKL = os.environ.get("GAME_MAP_PATH", "./pickle_files/game_map.pkl")
USER_MAP_PKL = os.environ.get("USER_MAP_PATH", "./pickle_files/user_map.pkl")
GAME_MAX_LENGTHS_PKL = os.environ.get("GAME_MAX_LENGTHS_PATH", "./pickle_files/game_max_lengths.pkl")
GAMES_CSV = os.environ.get("GAMES_CSV", "./csv_files/treated_dataframe.csv")

# Quantos resultados retornar por padrão
DEFAULT_TOP_K = 5

# Corpo dos campos
REQUIRED_FIELDS = [
    "lang_id",
    "top_genre",
    "top_cat",
    "top_tag",
    "genre_dominance",
    "genre_diversity"
]
# ------------------------------------------------------------------

app = FastAPI(title="DeepSeek TwoTower Recommender API")

# ---------- Request/Response models ----------
class UserProfile(BaseModel):
    # Esperamos que o profile já venha em ids numéricos compatíveis com o que o modelo usou.
    # Se você vai extrair do chat (texto), será necessário um mapeamento de texto -> id (ex.: 'action' -> genre_id 3).
    lang_id: Optional[int] = 0
    top_genre: Optional[int] = 0
    top_cat: Optional[int] = 0
    top_tag: Optional[int] = 0
    genre_dominance: Optional[float] = 0.0
    genre_diversity: Optional[float] = 0.0
    # Opcional: lista de appids para ranquear (se vazia, usamos pool padrão)
    candidate_appids: Optional[List[int]] = None
    top_k: Optional[int] = None

class RecoItem(BaseModel):
    appid: int
    score: float
    title: Optional[str] = None

class RecoResponse(BaseModel):
    user_profile: Dict[str, Any]
    top_k: int
    results: List[RecoItem]

# ---------- Utilitários ----------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _pad_list(lst, max_len, pad_value=0):
    if lst is None:
        lst = []
    if len(lst) >= max_len:
        return lst[:max_len]
    return lst + [pad_value] * (max_len - len(lst))

def game_matches_constraints(appid: int, profile: dict) -> bool:
    """
    Verifica as restrições obrigatórias:
      - top_genre ∈ game['genre_ids']
      - top_cat ∈ game['category_ids']
      - top_tag ∈ game['user_tag_ids']
      - game['review_score'] > 4.0
    Retorna False se algum dado do jogo estiver ausente/inválido.
    """
    gm = game_map.get(int(appid))
    if gm is None:
        return False

    # extrair listas (padrão: [])
    try:
        g_genres = gm.get('genre_ids', []) or []
        g_cats = gm.get('category_ids', []) or []
        g_tags = gm.get('user_tag_ids', []) or []
        rev = float(gm.get('review_score', 0.0))
    except Exception:
        # se qualquer conversão falhar, considerar incompatível
        return False

    # checar presença de ids (profile já contém ints)
    # Se top_genre/top_cat/top_tag forem 0 (padding / unknown) considerar que não satisfaz a restrição.
    tg = int(profile.get('top_genre', 0))
    tc = int(profile.get('top_cat', 0))
    tt = int(profile.get('top_tag', 0))

    if tg <= 0 or tc <= 0 or tt <= 0:
        # perfil inválido pra essas restrições — não permitir recomendação.
        return False

    if tg not in g_genres:
        return False
    if tc not in g_cats:
        return False
    if tt not in g_tags:
        return False
    if rev <= 4.0:
        return False

    return True

# ---------- Carregar artefatos (feito uma vez no startup) ----------
print("🔁 Carregando modelo e artefatos...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Modelo carregado.")

if not os.path.exists(GAME_MAP_PKL):
    raise FileNotFoundError(f"game_map.pkl não encontrado em {GAME_MAP_PKL}")
game_map = load_pickle(GAME_MAP_PKL)
print(f"✅ game_map carregado ({len(game_map)} jogos).")

if os.path.exists(USER_MAP_PKL):
    user_map = load_pickle(USER_MAP_PKL)
    print(f"✅ user_map carregado ({len(user_map)} usuários).")
else:
    user_map = {}
    print("⚠️ user_map não encontrado — ok para inferência a partir de user profile externo.")

if not os.path.exists(GAME_MAX_LENGTHS_PKL):
    raise FileNotFoundError(f"game_max_lengths.pkl não encontrado em {GAME_MAX_LENGTHS_PKL}")
game_max_lengths = load_pickle(GAME_MAX_LENGTHS_PKL)
print(f"✅ game_max_lengths: {game_max_lengths}")

# carregar games csv para metadata (título, etc.)
if os.path.exists(GAMES_CSV):
    games_df = pd.read_csv(GAMES_CSV, encoding='utf-8')
    # garantir que 'game_id' exista
    if 'game_id' not in games_df.columns:
        # tentar nome alternativo
        if 'gameId' in games_df.columns:
            games_df = games_df.rename(columns={'gameId': 'game_id'})
else:
    games_df = pd.DataFrame()
    print("⚠️ games csv não encontrado — retornaremos só appids e scores.")

# ---------- Função que monta inputs para modelo ----------
def build_inputs_for_user_and_candidates(user_profile: dict, candidate_appids: List[int]):
    """
    Constrói o dicionário de inputs requerido pelo modelo para inference:
    - user_* repetidos por len(candidate_appids)
    - game_* preenchidos conforme game_map e game_max_lengths
    """
    list_size = len(candidate_appids)
    # user inputs: arrays shape (list_size,) ou (list_size,1)
    u_lang = np.full((list_size,), int(user_profile.get('lang_id', 0)), dtype=np.int32)
    u_top_genre = np.full((list_size,), int(user_profile.get('top_genre', 0)), dtype=np.int32)
    u_top_cat = np.full((list_size,), int(user_profile.get('top_cat', 0)), dtype=np.int32)
    u_top_tag = np.full((list_size,), int(user_profile.get('top_tag', 0)), dtype=np.int32)
    u_genre_dom = np.full((list_size, 1), float(user_profile.get('genre_dominance', 0.0)), dtype=np.float32)
    u_genre_div = np.full((list_size, 1), float(user_profile.get('genre_diversity', 0.0)), dtype=np.float32)

    # game features
    g_genres = []
    g_cats = []
    g_tags = []
    g_reviews = []
    for appid in candidate_appids:
        gm = game_map.get(int(appid), None)
        if gm is None:
            # se não existe, usar padding arrays
            g_genres.append(_pad_list([], game_max_lengths['genre_ids']))
            g_cats.append(_pad_list([], game_max_lengths['category_ids']))
            g_tags.append(_pad_list([], game_max_lengths['user_tag_ids']))
            g_reviews.append([0.0])
        else:
            g_genres.append(_pad_list(gm.get('genre_ids', []), game_max_lengths['genre_ids']))
            g_cats.append(_pad_list(gm.get('category_ids', []), game_max_lengths['category_ids']))
            g_tags.append(_pad_list(gm.get('user_tag_ids', []), game_max_lengths['user_tag_ids']))
            g_reviews.append([float(gm.get('review_score', 0.0))])

    # Convert to numpy arrays with shapes expected pelo modelo:
    inputs = {
        'user_lang_id': np.expand_dims(u_lang, 0),              # (1, list_size)
        'user_top_genre': np.expand_dims(u_top_genre, 0),
        'user_top_cat': np.expand_dims(u_top_cat, 0),
        'user_top_tag': np.expand_dims(u_top_tag, 0),
        'user_genre_dominance': np.expand_dims(u_genre_dom, 0), # (1, list_size, 1)
        'user_genre_diversity': np.expand_dims(u_genre_div, 0),
        'game_genre_ids': np.expand_dims(np.array(g_genres, dtype=np.int32), 0),   # (1, list_size, max_genre_len)
        'game_category_ids': np.expand_dims(np.array(g_cats, dtype=np.int32), 0),
        'game_user_tag_ids': np.expand_dims(np.array(g_tags, dtype=np.int32), 0),
        'game_review_score': np.expand_dims(np.array(g_reviews, dtype=np.float32), 0) # (1, list_size, 1)
    }
    return inputs

# ---------- Endpoint principal ----------
@app.post("/recommend", response_model=RecoResponse)
def recommend(user_profile: UserProfile):
    profile = user_profile.dict()
    top_k = profile.get('top_k') or DEFAULT_TOP_K

    # Candidate pool: se o cliente mandou candidate_appids usa; caso contrário usa todos os jogos do game_map
    if profile.get('candidate_appids'):
        candidate_appids = [int(x) for x in profile['candidate_appids']]
    else:
        candidate_appids = list(game_map.keys())

    if len(candidate_appids) == 0:
        raise HTTPException(status_code=400, detail="Pool de candidatos vazio.")

    # ------------------ APLICAR RESTRIÇÕES OBRIGATÓRIAS ------------------
    # Filtrar os candidatos para manter somente os jogos que obedecem as regras do usuário
    filtered_candidates = []
    for aid in candidate_appids:
        try:
            if game_matches_constraints(aid, profile):
                filtered_candidates.append(int(aid))
        except Exception:
            # se houver exceção, ignorar esse appid
            continue

    # Se nenhum jogo satisfaz as restrições, retornar resultado vazio (você pode alterar para 400 se preferir)
    if not filtered_candidates:
        return RecoResponse(user_profile=profile, top_k=0, results=[])

    # Construir inputs e predizer usando somente os jogos filtrados
    inputs = build_inputs_for_user_and_candidates(profile, filtered_candidates)

    try:
        preds = model.predict(inputs, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao predizer com o modelo: {e}")

    # extrair scores como antes
    if isinstance(preds, dict):
        scores = preds.get('score')
    else:
        scores = preds

    if scores is None:
        raise HTTPException(status_code=500, detail="Modelo não retornou scores esperados.")

    scores = np.array(scores)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    elif scores.ndim == 1:
        pass
    else:
        scores = scores.reshape(-1)

    # Ordenar candidatos por score desc
    idx_order = np.argsort(scores)[::-1]
    top_idx = idx_order[:top_k]
    results = []
    for i in top_idx:
        idx_i = int(i)
        aid = int(filtered_candidates[idx_i])
        sc = float(scores[idx_i])
        title = None
        if not games_df.empty:
            row = games_df[games_df['game_id'] == aid]
            if not row.empty and 'title' in row.columns:
                title = str(row.iloc[0]['title'])
        results.append(RecoItem(appid=aid, score=sc, title=title))

    return RecoResponse(user_profile=profile, top_k=len(results), results=results)

@app.post("/check_profile")
def check_profile(profile: dict = Body(...)):
    missing = []
    invalid = {}
    for f in REQUIRED_FIELDS:
        if f not in profile:
            missing.append(f)
        else:
            v = profile[f]
            # validações simples
            if f in ["lang_id", "top_genre", "top_cat", "top_tag"]:
                if not isinstance(v, int):
                    invalid[f] = "deve ser integer (id). Se enviar rótulo, converta para id antes."
            if f == "genre_dominance":
                if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                    invalid[f] = "deve ser numérico entre 0.0 e 1.0"
            if f == "genre_diversity":
                if not isinstance(v, (int, float)):
                    invalid[f] = "deve ser numérico (ex: 1, 2, 1.5)"

    return {
        "missing_fields": missing,
        "invalid_fields": invalid,
        "all_good": (len(missing) == 0 and len(invalid) == 0)
    }

@app.post("/profile_from_chat", response_model=RecoResponse)
def profile_from_chat(profile: dict = Body(...)):
    # Validação básica
    check = check_profile(profile)
    if not check["all_good"]:
        raise HTTPException(status_code=400, detail={
            "message": "perfil incompleto/inválido",
            "check": check
        })

    # Construir payload do UserProfile pydantic que o endpoint recommend espera
    user_profile_payload = {
        "lang_id": int(profile["lang_id"]),
        "top_genre": int(profile["top_genre"]),
        "top_cat": int(profile["top_cat"]),
        "top_tag": int(profile["top_tag"]),
        "genre_dominance": float(profile["genre_dominance"]),
        "genre_diversity": float(profile["genre_diversity"]),
        "candidate_appids": profile.get("candidate_appids", None),
        "top_k": profile.get("top_k", None)
    }

    up = UserProfile(**user_profile_payload)
    return recommend(up)

# ---------- Health check ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "n_games": len(game_map)}

# ---------- Run (quando executar python backend.py) ----------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
