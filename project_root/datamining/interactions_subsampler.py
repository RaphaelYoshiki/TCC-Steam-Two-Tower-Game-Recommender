import pandas as pd

# Caminho do arquivo original
interactions_csv = "LLM/utils/all_data/cleaned_interactions_shuffled.csv"

# Arquivos de saída
val_csv = "LLM/utils/all_data/interactions_val.csv"
train_csv = "LLM/utils/all_data/interactions_train.csv"

# Fração para validação e semente para reprodução
VAL_FRAC = 0.20
RANDOM_STATE = 42

USER_COL = "steamid"  # altere para o nome da coluna de usuário no seu CSV, se diferente

df = pd.read_csv(interactions_csv)

if USER_COL in df.columns:
    users = df[USER_COL].unique()
    # sample users for val
    import numpy as np
    np.random.seed(RANDOM_STATE)
    val_user_count = int(len(users) * VAL_FRAC)
    val_users = set(np.random.choice(users, size=val_user_count, replace=False))

    val_by_user_df = df[df[USER_COL].isin(val_users)].reset_index(drop=True)
    train_by_user_df = df[~df[USER_COL].isin(val_users)].reset_index(drop=True)

    # salvar em arquivos diferentes para comparação (opcional)
    val_by_user_df.to_csv(val_csv, index=False)
    train_by_user_df.to_csv(train_csv, index=False)

    print(f"Split por usuário salvo: train={len(train_by_user_df)} linhas, val={len(val_by_user_df)} linhas.")
else:
    print(f"Coluna '{USER_COL}' não encontrada.")