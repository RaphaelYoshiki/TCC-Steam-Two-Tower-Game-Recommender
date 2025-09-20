import pandas as pd
import numpy as np
from pathlib import Path
import os

def clean_user_profiles(user_profiles_file, games_base_file, output_file):
    """
    Limpa os perfis de usuário removendo duplicatas e apps não existentes na base de jogos
    
    Args:
        user_profiles_file (str): Arquivo CSV com os perfis de usuário extraídos
        games_base_file (str): Arquivo CSV com a base de jogos (treated_dataframe.csv)
        output_file (str): Arquivo de saída limpo
    """
    
    print("Carregando dados...")
    
    # Carregar perfis de usuário
    user_profiles_df = pd.read_csv(user_profiles_file)
    print(f"Perfis de usuário carregados: {len(user_profiles_df)} linhas")
    
    # Carregar base de jogos
    games_df = pd.read_csv(games_base_file)
    print(f"Base de jogos carregada: {len(games_df)} jogos")
    
    # Extrair lista de appids válidos da base de jogos
    valid_appids = set(games_df['game_id'].astype(int).unique())
    print(f"AppIDs válidos na base: {len(valid_appids)}")
    
    # 1. Remover linhas duplicadas (todas as colunas iguais)
    before_dedup = len(user_profiles_df)
    user_profiles_df = user_profiles_df.drop_duplicates()
    after_dedup = len(user_profiles_df)
    print(f"Duplicatas removidas: {before_dedup - after_dedup} linhas")
    
    # 2. Remover linhas com appid que não existe na base de jogos
    before_filter = len(user_profiles_df)
    user_profiles_df = user_profiles_df[user_profiles_df['appid'].isin(valid_appids)]
    after_filter = len(user_profiles_df)
    print(f"Linhas com appid inválido removidas: {before_filter - after_filter}")
    
    # 3. Remover linhas com valores críticos missing
    # Verificar e remover linhas com steamid ou appid missing
    before_missing = len(user_profiles_df)
    user_profiles_df = user_profiles_df.dropna(subset=['steamid', 'appid'])
    after_missing = len(user_profiles_df)
    print(f"Linhas com steamid ou appid missing removidas: {before_missing - after_missing}")
    
    # 4. Limpar dados textuais (reviews)
    # Remover reviews vazias ou apenas whitespace
    if 'review' in user_profiles_df.columns:
        before_review = len(user_profiles_df)
        user_profiles_df = user_profiles_df[user_profiles_df['review'].notna()]
        user_profiles_df = user_profiles_df[user_profiles_df['review'].str.strip() != '']
        after_review = len(user_profiles_df)
        print(f"Linhas com review vazia removidas: {before_review - after_review}")
    
    # 5. Converter tipos de dados
    # Garantir que appid seja inteiro
    user_profiles_df['appid'] = user_profiles_df['appid'].astype(int)
    
    # Garantir que playtime_forever seja numérico
    if 'playtime_forever' in user_profiles_df.columns:
        user_profiles_df['playtime_forever'] = pd.to_numeric(user_profiles_df['playtime_forever'], errors='coerce').fillna(0)
    
    # 6. Ordenar por timestamp (se existir) para análise temporal
    if 'timestamp_created' in user_profiles_df.columns:
        user_profiles_df = user_profiles_df.sort_values('timestamp_created', ascending=False)
    
    # 7. Salvar arquivo limpo
    print(user_profiles_df.head(10))
    print(user_profiles_df.tail(10))
    user_profiles_df.sample(frac=1).reset_index(drop=True).to_csv(output_file, index=False)
    print(f"Arquivo limpo salvo: {output_file}")
    print(f"Total de linhas no arquivo final: {len(user_profiles_df)}")
    
    # Estatísticas finais
    print("\n=== ESTATÍSTICAS FINAIS ===")
    print(f"Perfis únicos de usuário: {user_profiles_df['steamid'].nunique()}")
    print(f"Jogos únicos com reviews: {user_profiles_df['appid'].nunique()}")
    
    if 'voted_up' in user_profiles_df.columns:
        positive_reviews = user_profiles_df['voted_up'].sum()
        negative_reviews = len(user_profiles_df) - positive_reviews
        print(f"Reviews positivas: {positive_reviews} ({positive_reviews/len(user_profiles_df)*100:.1f}%)")
        print(f"Reviews negativas: {negative_reviews} ({negative_reviews/len(user_profiles_df)*100:.1f}%)")
    
    if 'playtime_forever' in user_profiles_df.columns:
        avg_playtime = user_profiles_df['playtime_forever'].mean()
        print(f"Tempo médio de jogo: {avg_playtime:.1f} minutos")
    
    return user_profiles_df

def analyze_user_activity(user_profiles_df):
    """
    Analisa a atividade dos usuários
    """
    print("\n=== ANÁLISE DE ATIVIDADE DOS USUÁRIOS ===")
    
    # Contar reviews por usuário
    reviews_per_user = user_profiles_df['steamid'].value_counts()
    print(f"Média de reviews por usuário: {reviews_per_user.mean():.2f}")
    print(f"Mediana de reviews por usuário: {reviews_per_user.median():.2f}")
    print(f"Usuário mais ativo: {reviews_per_user.max()} reviews")
    
    # Top 10 usuários mais ativos
    print("\nTop 10 usuários mais ativos:")
    for i, (user_id, count) in enumerate(reviews_per_user.head(10).items(), 1):
        print(f"{i}. {user_id}: {count} reviews")
    
    # Reviews por jogo
    reviews_per_game = user_profiles_df['appid'].value_counts()
    print(f"\nMédia de reviews por jogo: {reviews_per_game.mean():.2f}")
    print(f"Jogo mais revisado: {reviews_per_game.max()} reviews")
    
    # Top 10 jogos mais revisados
    print("\nTop 10 jogos mais revisados:")
    for i, (game_id, count) in enumerate(reviews_per_game.head(10).items(), 1):
        print(f"{i}. {game_id}: {count} reviews")

if __name__ == "__main__":
    # Configurações
    USER_PROFILES_FILE = "LLM/utils/all_data/interactions.csv"
    GAMES_BASE_FILE = "LLM/utils/all_data/treated_dataframe.csv"
    OUTPUT_FILE = "LLM/utils/all_data/cleaned_interactions_shuffled.csv"
    
    # Executar limpeza
    try:
        cleaned_df = clean_user_profiles(USER_PROFILES_FILE, GAMES_BASE_FILE, OUTPUT_FILE)
        
        # Opcional: Análise adicional
        analyze_user_activity(cleaned_df)
        
        print(f"\nProcessamento concluído! Arquivo salvo como: {OUTPUT_FILE}")
        
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()