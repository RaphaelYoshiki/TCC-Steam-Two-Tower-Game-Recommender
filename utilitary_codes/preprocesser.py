import json
import pickle
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Carregar listas de valores válidos
with open('gamedata_value_lists/developers_list.json', encoding='utf-8') as f:
    valid_devs = json.load(f)
with open('gamedata_value_lists/genres_list.json', encoding='utf-8') as f:
    valid_genres = json.load(f)

# Carregar dados dos jogos
games_data = {}
with open(f'filtered_games/mixed_normalized.p', 'rb') as m:
    games_data.update(pickle.load(m))
with open(f'filtered_games/mostly_positive_normalized.p', 'rb') as mp:
    games_data.update(pickle.load(mp))
with open(f'filtered_games/overwhelmingly_positive_normalized.p', 'rb') as op:
    games_data.update(pickle.load(op))
with open(f'filtered_games/very_positive_normalized.p', 'rb') as vp:
    games_data.update(pickle.load(vp))

# Pré-processamento
def preprocess_game(game):
    features = {}
    
    # Codificar desenvolvedores (multi-hot)
    devs = [d for d in game.get('developers', []) if d in valid_devs]
    features['developers'] = [1 if dev in devs else 0 for dev in valid_devs]
    
    # Codificar gêneros (multi-hot)
    genres = [g['description'] for g in game.get('genres', []) if g['description'] in valid_genres]
    features['genres'] = [1 if genre in genres else 0 for genre in valid_genres]
    
    # Preço (normalizado)
    price = game.get('price_overview', {}).get('final', 0) / 10000  # Normalizar
    features['price'] = max(0, min(1, price))  # Clip entre 0 e 1
    
    # Plataformas (multi-hot)
    platforms = game.get('platforms', {})
    features['platforms'] = [
        int(platforms.get('windows', False)),
        int(platforms.get('mac', False)),
        int(platforms.get('linux', False))
    ]
    
    # Avaliação (codificar texto para numérico)
    review_map = {
        'Overwhelmingly Positive': 5,
        'Very Positive': 4,
        'Positive': 3,
        'Mostly Positive': 2,
        'Mixed': 1,
        'Negative': 0
    }
    features['review_score'] = review_map.get(game.get('review_score', 'Mixed'), 1) / 5
    
    # Data de lançamento (extrair ano e normalizar)
    release_date = game.get('release_date', {}).get('date', '')
    year = int(release_date.split(', ')[-1]) if release_date and ',' in release_date else 2000
    features['release_year'] = (year - 2000) / 25  # Normalizar entre 0 e 1 para anos 2000-2025
    
    return features

# Criar dataset processado
processed_games = {appid: preprocess_game(data) for appid, data in games_data.items()}