import pickle
import tensorflow as tf

MODEL_PATH = "./models/two_tower_realusers_26.keras"
PICKLE_DIR = "./pickle_files"

def load_pickle(name):
    with open(f"{PICKLE_DIR}/{name}", "rb") as f:
        return pickle.load(f)

def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    game_map = load_pickle("game_map.pkl")
    game_max_lengths = load_pickle("game_max_lengths.pkl")

    return model, game_map, game_max_lengths
