import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def clean_user_profiles(user_profiles_file, games_base_file, output_file):
    """
    Limpa os perfis de usuário removendo duplicatas, apps inexistentes e outliers de atividade.
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
    
    # 1. Remover duplicatas
    before_dedup = len(user_profiles_df)
    user_profiles_df = user_profiles_df.drop_duplicates()
    print(f"Duplicatas removidas: {before_dedup - len(user_profiles_df)}")
    
    # 2. Remover appids inválidos
    before_filter = len(user_profiles_df)
    user_profiles_df = user_profiles_df[user_profiles_df['appid'].isin(valid_appids)]
    print(f"Linhas com appid inválido removidas: {before_filter - len(user_profiles_df)}")
    
    # 3. Remover valores críticos ausentes
    before_missing = len(user_profiles_df)
    user_profiles_df = user_profiles_df.dropna(subset=['steamid', 'appid'])
    print(f"Linhas com steamid/appid ausentes removidas: {before_missing - len(user_profiles_df)}")
    
    # 4. Limpar reviews vazias
    if 'review' in user_profiles_df.columns:
        before_review = len(user_profiles_df)
        user_profiles_df = user_profiles_df[user_profiles_df['review'].notna()]
        user_profiles_df = user_profiles_df[user_profiles_df['review'].str.strip() != '']
        print(f"Linhas com review vazia removidas: {before_review - len(user_profiles_df)}")
    
    # 5. Converter tipos
    user_profiles_df['appid'] = user_profiles_df['appid'].astype(int)
    if 'playtime_forever' in user_profiles_df.columns:
        user_profiles_df['playtime_forever'] = pd.to_numeric(
            user_profiles_df['playtime_forever'], errors='coerce'
        ).fillna(0)
    
    # 6. Ordenar por timestamp (se existir)
    if 'timestamp_created' in user_profiles_df.columns:
        user_profiles_df = user_profiles_df.sort_values('timestamp_created', ascending=False)
    
    # 7. Calcular estatísticas de atividade dos usuários
    print("\nCalculando estatísticas de atividade dos usuários...")
    reviews_per_user = user_profiles_df['steamid'].value_counts()
    mean_reviews = reviews_per_user.mean()
    std_reviews = reviews_per_user.std()
    
    print(f"Média de reviews por usuário: {mean_reviews:.2f}")
    print(f"Desvio padrão: {std_reviews:.2f}")
    
    # 8. Gerar gráfico de torre
    plt.figure(figsize=(10,6))
    count_freq = reviews_per_user.value_counts().sort_index()
    plt.bar(count_freq.index, count_freq.values, width=1.0)
    plt.title("Distribuição de número de reviews por usuário")
    plt.xlabel("Número de reviews por usuário")
    plt.ylabel("Número de usuários")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    tower_path = os.path.splitext(output_file)[0] + "_review_tower.png"
    plt.savefig(tower_path, dpi=300)
    plt.close()
    print(f"Gráfico de torre salvo em: {tower_path}")
    
    # 9. Remover usuários fora da faixa média ± desvio padrão
    lower_limit = 15
    upper_limit = 100000000000
    print(f"Removendo usuários com menos de {lower_limit:.2f} ou mais de {upper_limit:.2f} reviews...")
    
    valid_users = reviews_per_user[
        (reviews_per_user >= lower_limit) & (reviews_per_user <= upper_limit)
    ].index
    
    before_outlier_filter = len(user_profiles_df)
    user_profiles_df = user_profiles_df[user_profiles_df['steamid'].isin(valid_users)]
    print(f"Linhas removidas (outliers): {before_outlier_filter - len(user_profiles_df)}")
    
    # 10. Salvar arquivo final
    user_profiles_df.sample(frac=1).reset_index(drop=True).to_csv(output_file, index=False)
    print(f"\nArquivo final salvo: {output_file}")
    print(f"Total de linhas no arquivo final: {len(user_profiles_df)}")
    print(f"Usuários únicos restantes: {user_profiles_df['steamid'].nunique()}")

    # 11. Separar 80% dos usuários para treino e 20% para restante
    print("\nSeparando 80% dos usuários para treino...")
    unique_users = user_profiles_df['steamid'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)

    split_idx = int(len(unique_users) * 0.8)
    train_users = set(unique_users[:split_idx])
    rest_users = set(unique_users[split_idx:])

    df_train = user_profiles_df[user_profiles_df['steamid'].isin(train_users)]
    df_rest = user_profiles_df[user_profiles_df['steamid'].isin(rest_users)]

    train_file = os.path.splitext(output_file)[0] + "_train_80.csv"
    rest_file = os.path.splitext(output_file)[0] + "_rest_20.csv"

    df_train.to_csv(train_file, index=False)
    df_rest.to_csv(rest_file, index=False)

    print(f"Treino: {len(df_train)} linhas ({len(train_users)} usuários)")
    print(f"Restante: {len(df_rest)} linhas ({len(rest_users)} usuários)")
    print(f"Arquivos salvos:\n- {train_file}\n- {rest_file}")
    
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