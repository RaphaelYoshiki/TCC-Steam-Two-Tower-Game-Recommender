# two_tower_training.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import tensorflow_ranking as tfr
import pandas as pd

from LLM.utils.data_prep import load_and_preprocess_interactions
from two_tower_model import create_two_tower_model

# ---------------- CONFIGURAÇÕES ----------------
TRAIN_NUM = 2
TRAIN_CSV = 'LLM/utils/all_data/interactions_train.csv'
VAL_CSV = 'LLM/utils/all_data/interactions_val.csv'
GAMES_CSV = 'LLM/utils/all_data/treated_dataframe.csv'
BATCH_SIZE = 1024
EMBEDDING_DIM = 128
CHUNK_SIZE = 250_000
EPOCHS_PER_CHUNK = 5
MODEL_SAVE_PATH = f'LLM/utils/all_data/models/gr_two_tower_{TRAIN_NUM}.keras'
LOSS_PLOT = f'LLM/utils/all_data/model_plots/loss_plot_{TRAIN_NUM}.png'
MAP_PLOT = f'LLM/utils/all_data/model_plots/map_plot_{TRAIN_NUM}.png'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOSS_PLOT), exist_ok=True)
os.makedirs(os.path.dirname(MAP_PLOT), exist_ok=True)
NEG_K = 10
# ------------------------------------------------

class GlobalEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.best_metric = -np.Inf
        self.best_weights = None
        self.wait = 0
        self.stop_training_flag = False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_map_metric')
        if current is None:
            return
        if current > self.best_metric:
            self.best_metric = current
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.stop_training_flag = True
                print(f"Global early stopping triggered at epoch {epoch}.")

    def restore_best_weights(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print("Restored globally best weights.")

class BPRListLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        pos_scores = y_pred[:, 0]          # (batch,)
        neg_scores = y_pred[:, 1:]         # (batch, k)
        diff = tf.expand_dims(pos_scores, 1) - neg_scores
        return -tf.reduce_mean(tf.math.log_sigmoid(diff))

def main():
    print("Inicializando (carregando primeiro chunk de treino para shapes)...")
    first_reader = pd.read_csv(TRAIN_CSV, chunksize=CHUNK_SIZE)
    first_chunk_df = next(first_reader)
    temp_csv = "temp_chunk_init.csv"
    first_chunk_df.to_csv(temp_csv, index=False)

    ds_init, user_max_lengths, game_max_lengths, max_values = load_and_preprocess_interactions(
        temp_csv, GAMES_CSV, batch_size=BATCH_SIZE, neg_k=NEG_K
    )

    # carregar val fixo (todo o arquivo)
    val_ds, _, _, _ = load_and_preprocess_interactions(
        VAL_CSV, GAMES_CSV, batch_size=BATCH_SIZE, neg_k=NEG_K
    )

    model = create_two_tower_model(
        user_max_lengths=user_max_lengths,
        game_max_lengths=game_max_lengths,
        embedding_dim=EMBEDDING_DIM,
        unique_game_devs=max_values['unique_game_devs'],
        unique_game_gens=max_values['unique_game_gens'],
        unique_game_cats=max_values['unique_game_cats'],
        unique_game_tags=max_values['unique_game_tags']
    )

    map_metric = tfr.keras.metrics.MeanAveragePrecisionMetric(name="map_metric", topn=NEG_K+1)
    optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=BPRListLoss(), metrics=[map_metric])

    ges = GlobalEarlyStopping(patience=3)

    all_loss, all_map, all_val_loss, all_val_map = [], [], [], []

    print("Treinando no primeiro chunk...")
    history = model.fit(ds_init,
                        validation_data=val_ds,
                        epochs=EPOCHS_PER_CHUNK,
                        callbacks=[ges])
    all_loss.extend(history.history['loss'])
    all_map.extend(history.history['map_metric'])
    all_val_loss.extend(history.history['val_loss'])
    all_val_map.extend(history.history['val_map_metric'])

    initial_epoch = EPOCHS_PER_CHUNK
    reader = pd.read_csv(TRAIN_CSV, chunksize=CHUNK_SIZE, skiprows=range(1, CHUNK_SIZE+1), header=0)
    for i, chunk_df in enumerate(reader, start=2):
        if ges.stop_training_flag:
            print("Parando o treinamento.")
            break
        print(f"--- Treinando no chunk {i} com {len(chunk_df)} interações ---")
        temp_csv = f"temp_chunk_{i}.csv"
        chunk_df.to_csv(temp_csv, index=False)
        ds_chunk, _, _, _ = load_and_preprocess_interactions(
            temp_csv, GAMES_CSV, batch_size=BATCH_SIZE, neg_k=NEG_K
        )

        history = model.fit(ds_chunk,
                            validation_data=val_ds,
                            epochs=initial_epoch + EPOCHS_PER_CHUNK,
                            initial_epoch=initial_epoch,
                            callbacks=[ges])
        all_loss.extend(history.history['loss'])
        all_map.extend(history.history['map_metric'])
        all_val_loss.extend(history.history['val_loss'])
        all_val_map.extend(history.history['val_map_metric'])
        initial_epoch += EPOCHS_PER_CHUNK
        
    ges.restore_best_weights()

    print("Treino finalizado. Salvando modelo em", MODEL_SAVE_PATH)
    model.save(MODEL_SAVE_PATH)

    print("Salvando imagens e gráficos...")
    epochs = range(1, len(all_loss) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, all_loss, label='Train Loss')
    plt.plot(epochs, all_val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss x Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PLOT)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, all_map, label='Train MAP@k')
    plt.plot(epochs, all_val_map, label='Val MAP@k')
    plt.xlabel('Epoch')
    plt.ylabel('MAP')
    plt.title('MAP x Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(MAP_PLOT)

    print(f"Gráficos salvos em {LOSS_PLOT} e {MAP_PLOT}")

if __name__ == '__main__':
    main()
