# Este código acessa a API da Steam para extrair perfis de usuário a partir das reviews dos jogos
# Utiliza a lista de IDs excluídos do código de extração de dados de jogos

from datetime import datetime
import os
import time
import requests
import json
import pickle
from pathlib import Path
import traceback
import csv
from collections import deque

def print_log(*args):
    print(f"[{str(datetime.now())[:-3]}] ", end="")
    print(*args)
    
def save_pickle(path_to_save:Path, obj):
    with open(path_to_save, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_excluded_apps():
    """Carrega a lista de apps excluídos do checkpoint"""
    checkpoint_folder = Path('game_data_checkpoints').resolve()
    exc_apps_filename_prefix = 'excluded_apps_list-ckpt-fin'
    
    # Buscar o checkpoint mais recente de apps excluídos
    all_pkl = []
    for root, dirs, files in os.walk(checkpoint_folder):
        all_pkl = list(map(lambda f: Path(root, f), files))
        all_pkl = [p for p in all_pkl if p.suffix == '.p']
        break
    
    exc_apps_ckpt_files = [f for f in all_pkl if exc_apps_filename_prefix in f.name and "ckpt" in f.name]
    exc_apps_ckpt_files.sort()
    
    if exc_apps_ckpt_files:
        latest_exc_apps_ckpt_path = exc_apps_ckpt_files[-1]
        excluded_apps_list = pickle.load(open(latest_exc_apps_ckpt_path, "rb"))
        print_log(f'Carregada lista de {len(excluded_apps_list)} apps excluídos')
        return set(excluded_apps_list)
    else:
        print_log('Nenhum checkpoint de apps excluídos encontrado')
        return set()
    
def load_checkpoint():    
    """Carrega a lista de apps excluídos do checkpoint"""
    checkpoint_folder = Path('user_data_checkpoints').resolve()
    exc_apps_filename_prefix = 'user_interactions_ckpt'
    
    # Buscar o checkpoint mais recente de apps excluídos
    all_pkl = []
    for root, dirs, files in os.walk(checkpoint_folder):
        all_pkl = list(map(lambda f: Path(root, f), files))
        all_pkl = [p for p in all_pkl if p.suffix == '.p']
        break
    
    exc_apps_ckpt_files = [f for f in all_pkl if exc_apps_filename_prefix in f.name and "ckpt" in f.name]
    exc_apps_ckpt_files.sort()
    
    if exc_apps_ckpt_files:
        latest_exc_apps_ckpt_path = exc_apps_ckpt_files[-1]
        excluded_apps_list = pickle.load(open(latest_exc_apps_ckpt_path, "rb"))
        print_log(f'Carregada lista de {len(excluded_apps_list)} apps excluídos')
        return set(excluded_apps_list)
    else:
        print_log('Nenhum checkpoint de apps excluídos encontrado')
        return set()

def get_all_app_ids():
    """Obtém todos os IDs de apps da Steam"""
    req = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/")

    if req.status_code != 200:
        print_log("Falha ao obter lista de jogos da Steam")
        return []

    try:
        data = req.json()
    except Exception as e:
        traceback.print_exc(limit=5)
        return []

    apps_data = data['applist']['apps']
    apps_ids = []

    for app in apps_data:
        appid = app['appid']
        name = app['name']
        
        # Pular apps com nome vazio
        if not name:
            continue

        apps_ids.append(appid)

    return apps_ids

def get_app_reviews(appid, cursor="*", num_per_page=100):
    """Obtém reviews de um app específico"""
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        'json': 1,
        'filter': 'all',
        'language': 'all',
        'day_range': '365',
        'cursor': cursor,
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': num_per_page,
        'filter_offtopic_activity': 0
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print_log(f"Obtido reviews de app {appid}, página de cursor {cursor}.")
            return response.json()
        elif response.status_code == 429:
            print_log(f"Muitas requisições para app {appid}. Aguardando...")
            time.sleep(10)
            return get_app_reviews(appid, cursor, num_per_page)
        elif response.status_code == 403:
            print_log(f"Acesso proibido para app {appid}. Aguardando 5 minutos...")
            time.sleep(300)
            return get_app_reviews(appid, cursor, num_per_page)
        else:
            print_log(f"Erro {response.status_code} para app {appid}")
            return None
    except Exception as e:
        print_log(f"Exceção ao buscar reviews do app {appid}: {e}")
        return None

def extract_user_profiles_from_reviews(reviews_data, appid):
    """Extrai perfis de usuário das reviews"""
    user_profiles = []
    
    if not reviews_data or 'reviews' not in reviews_data:
        return user_profiles
    
    for review in reviews_data['reviews']:
        try:
            user_profile = {
                'steamid': review['author']['steamid'],
                'appid': appid,
                'playtime_forever': review['author'].get('playtime_forever', 0),
                'language': review.get('language', 'unknown'),
                'voted_up': review.get('voted_up', False),
                'steam_purchase': review.get('steam_purchase', False)
            }
            user_profiles.append(user_profile)
        except KeyError as e:
            print_log(f"Chave faltando na review do app {appid}: {e}")
            continue
    
    return user_profiles

def save_user_profiles_to_csv(user_profiles, output_file):
    """Salva perfis de usuário em arquivo CSV"""
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['steamid', 'appid', 'playtime_forever', 'language', 
                     'voted_up', 'steam_purchase']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for profile in user_profiles:
            writer.writerow(profile)

def main():
    print_log("Iniciando extração de perfis de usuário")
    
    # Filepaths
    checkpoint_folder = Path('user_data_checkpoints').resolve()
    print_log('Checkpoint folder:', checkpoint_folder)
    if not checkpoint_folder.exists():
        print_log(f'Fail to find checkpoint folder: {checkpoint_folder}')
        print_log(f'Start at blank.')
        checkpoint_folder.mkdir(parents=True)
    
    output_file = "LLM/utils/all_data/interactions.csv"
    
    # Carregar apps excluídos
    excluded_apps = load_excluded_apps()
    print_log(f"Apps excluídos carregados: {len(excluded_apps)}")
    
    # Carregar apps já processados
    processed_apps = load_checkpoint()
    processed_appids = []
    processed_appids = [appid for appid in processed_apps]
    print_log(f"Apps já processados carregados: {len(processed_apps)}")
    
    # Obter todos os IDs de apps
    all_app_ids = get_all_app_ids()
    print_log(f"Total de apps na Steam: {len(all_app_ids)}")
    
    # Filtrar apps excluídos
    valid_app_ids = [appid for appid in all_app_ids if appid not in excluded_apps and appid not in processed_apps]
    print_log(f"Apps válidos para processamento: {len(valid_app_ids)}")
    
    # Processar apps
    processed_count = 0
    error_count = 0
    
    for appid in valid_app_ids:
        print_log(f"Processando app {appid} ({processed_count + 1}/{len(valid_app_ids)})")
        
        cursor = "*"
        has_more_reviews = True
        app_user_profiles = []
        
        limiter = 0
        
        while has_more_reviews and limiter < 5:
            reviews_data = get_app_reviews(appid, cursor)
            
            if not reviews_data:
                error_count += 1
                break
            
            if reviews_data.get('success', 0) != 1:
                print_log(f"Resposta não bem-sucedida para app {appid}")
                error_count += 1
                break
            
            # Extrair perfis desta página de reviews
            user_profiles_batch = extract_user_profiles_from_reviews(reviews_data, appid)
            app_user_profiles.extend(user_profiles_batch)
            
            # Verificar se há mais páginas
            cursor = reviews_data.get('cursor')
            has_more_reviews = cursor is not None and cursor != "*"
            
            limiter += 1
            time.sleep(0.1)
            
        processed_appids.append(appid)
        
        # Salvar perfis deste app
        if app_user_profiles:
            save_user_profiles_to_csv(app_user_profiles, output_file)
            print_log(f"Salvos {len(app_user_profiles)} interações de usuários com o app {appid}")
        
        processed_count += 1
        
        # Salvar checkpoint a cada 50 apps
        if processed_count % 50 == 0:
            save_pickle(checkpoint_folder.joinpath('user_interactions_ckpt.p').resolve(), processed_appids)
            print_log(f"Checkpoint: {processed_count} apps processados, {error_count} erros")
        
        # Pausa maior a cada 10 apps para evitar rate limiting
        if processed_count % 10 == 0:
            time.sleep(2)
    
    print_log("Extração concluída!")
    print_log(f"Total de apps processados: {processed_count}")
    print_log(f"Total de erros: {error_count}")
    print_log(f"Perfis salvos em: {output_file}")

if __name__ == '__main__':
    main()