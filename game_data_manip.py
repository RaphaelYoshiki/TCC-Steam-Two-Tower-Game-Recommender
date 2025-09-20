import pandas as pd
import ast
import json
import datetime
import ast
from dateparser import parse
from collections import defaultdict, OrderedDict

review_score_map = {
    'Overwhelmingly Positive': 8,
    'Very Positive': 7,
    'Positive': 6,
    'Mostly Positive': 5,
    'Mixed': 4,
    'Mostly Negative': 3,
    'Negative': 2,
    'Very Negative': 1,
    'Overwhelmingly Negative': 0
}

# Dicionários para mapear strings para IDs
developers_map = {}
genres_map = {}
categories_map = {}
user_tags_map = {}
platforms_map = {'windows': 1, 'mac': 2, 'linux': 3}  # IDs fixos para plataformas

# Contadores para IDs
dev_id_counter = 1
tag_id_counter = 1

def update_genre_id(gen_name, gen_id):
    if gen_id not in genres_map:
        genres_map[gen_id] = gen_name

def update_category_id(cat_name, cat_id):
    if cat_id not in categories_map:
        categories_map[cat_id] = cat_name

def get_developer_id(dev_name):
    global dev_id_counter
    if dev_name not in developers_map:
        developers_map[dev_name] = dev_id_counter
        dev_id_counter += 1
    return developers_map[dev_name]

def get_tag_id(tag_name):
    global tag_id_counter
    if tag_name not in user_tags_map:
        user_tags_map[tag_name] = tag_id_counter
        tag_id_counter += 1
    return user_tags_map[tag_name]

# Importando Dados
df = pd.read_json('filtered_datasets/normalized_filtered_dataset.json')

## ----- MANIPULAÇÃO E CONCATENAÇÃO DE DADOS ----- ##
extracted_data = []

for game_id in df.columns:
    game_info = df[game_id]
    
    name = game_info.get('name', [])
    
    # Processar developers com IDs
    devs = game_info.get('developers', [])
    dev_ids = [get_developer_id(dev) for dev in devs] if devs else []

    # Processar categories com IDs
    categories = game_info.get('categories', [])
    category_ids = []
    for cat in categories:
        if isinstance(cat, dict):
            cat_id = cat.get('id')
            cat_name = cat.get('description')
            if cat_id is not None:
                update_category_id(cat_name, cat_id)
                category_ids.append(cat_id)

    # Processar genres com IDs
    genres = game_info.get('genres', [])
    genre_ids = []
    for genre in genres:
        if isinstance(genre, dict):
            genre_id = ast.literal_eval(genre.get('id'))
            genre_name = genre.get('description')
            if genre_id is not None:
                update_genre_id(genre_name, genre_id)
                genre_ids.append(genre_id)

    review_score_text = game_info.get('review_score')
    review_score_numeric = review_score_map.get(review_score_text, 0)
    
    user_tags = game_info.get('user_tags', {})
    user_tag_ids = [get_tag_id(tag) for tag in user_tags.keys()] if user_tags else []

    extracted_data.append({
        'game_id': game_id,
        'name': name,
        'developer_ids': dev_ids,
        'category_ids': category_ids,
        'genre_ids': genre_ids,
        'review_score': review_score_numeric,
        'user_tag_ids': user_tag_ids
    })

df_extracted = pd.DataFrame(extracted_data)

# Tratamento de Dados
print("Valores nulos por coluna:")
print(df_extracted.isna().sum())

# Eliminando Linhas com Valores Inválidos
empty_lists = df_extracted.apply(
    lambda row: len(row['developer_ids']) == 0
    or len(row['category_ids']) == 0
    or len(row['genre_ids']) == 0,
    axis=1
)

df_extracted_treated = df_extracted[~empty_lists].dropna()

# Substituir listas vazias por ['Unknown'] apenas para developers e publishers
df_extracted_treated['developer_ids'] = df_extracted_treated['developer_ids'].apply(lambda x: [0] if len(x) == 0 else x)

# Dicionário para armazenar os formatos encontrados
date_formats = defaultdict(int)

# Ordenar dicts de genres e categories crescentemente por id
genres_map = OrderedDict(sorted(genres_map.items()))
categories_map = OrderedDict(sorted(categories_map.items()))

# Salvar os dicionários de mapeamento
mappings = {
    'developers': {v: k for k, v in developers_map.items()},
    'genres': genres_map,
    'categories': categories_map,
    'user_tags': {v: k for k, v in user_tags_map.items()},
    'platforms': platforms_map
}

with open('./LLM/utils/all_data/id_mappings.json', 'w', encoding='utf-8') as f:
    json.dump(mappings, f, indent=4, ensure_ascii=False)

# Salvar o dataframe tratado
dataframe_name = './LLM/utils/all_data/treated_dataframe.csv'
df_extracted_treated.to_csv(dataframe_name, index_label='index', index=True)
print(f'Saved dataframe to {dataframe_name}')
print(f'Saved ID mappings to id_mappings.json')