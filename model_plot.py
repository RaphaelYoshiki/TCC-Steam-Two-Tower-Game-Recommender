# plot_saved_model_custom.py
import os
import sys
import traceback
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt

# === IMPORTS dos objetos customizados ===
# Ajuste o caminho para onde seu two_tower_training.py está.
# Exemplo: se estiver no mesmo diretório, apenas:
from two_tower_training import BPRListLoss  # sua classe customizada
import tensorflow_ranking as tfr
MeanAveragePrecisionMetric = tfr.keras.metrics.MeanAveragePrecisionMetric

MODEL_PATH = r'LLM/utils/all_data/gr_two_tower.keras'  # ajuste se necessário
OUT_DIR = r'D:/Users/raphakun1010/Documents/Raphael/UFF/TCC/TCC - Sistema de Recomendação/model_plots'
OUT_IMG = 'model.png'
SUMMARY_TXT = 'model_summary.txt'
LAYER_PLOT = 'model_layers_plot.png'

def ensure_out_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_model_summary(model, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda s: f.write(s + '\n'))
    print(f"Model summary saved to: {filepath}")

def simple_layers_plot(model, filepath):
    layer_names = []
    param_counts = []
    for layer in model.layers:
        layer_names.append(layer.name)
        try:
            param_counts.append(layer.count_params())
        except Exception:
            param_counts.append(0)

    # agrupar se muitas camadas
    if len(layer_names) > 80:
        names, params = [], []
        block = 5
        for i in range(0, len(layer_names), block):
            names.append(f'{i+1}-{min(i+block, len(layer_names))}')
            params.append(sum(param_counts[i:i+block]))
        layer_names, param_counts = names, params

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(param_counts)), param_counts)
    plt.xticks(range(len(param_counts)), layer_names, rotation=90, fontsize=8)
    plt.ylabel('Parameter count')
    plt.title('Parameters per layer (grouped if many layers)')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Layer-parameters plot saved to: {filepath}")

def try_plot_model(model, out_path):
    try:
        print("Trying tf.keras.utils.plot_model(...)")
        plot_model(model, to_file=out_path, show_shapes=True)
        print(f"Model plot saved to: {out_path}")
        return True
    except Exception:
        print("plot_model failed with exception:")
        traceback.print_exc()
        return False

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model path not found: {MODEL_PATH}", file=sys.stderr)
        return

    ensure_out_dir(OUT_DIR)
    out_img_path = os.path.join(OUT_DIR, OUT_IMG)
    summary_path = os.path.join(OUT_DIR, SUMMARY_TXT)
    layer_plot_path = os.path.join(OUT_DIR, LAYER_PLOT)

    print("Loading model from:", MODEL_PATH)
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,  # evita recompilar
            custom_objects={
                'BPRListLoss': BPRListLoss,
                'MeanAveragePrecisionMetric': MeanAveragePrecisionMetric
            }
        )
    except Exception:
        print("Failed to load model. Exception:")
        traceback.print_exc()
        return

    ok = try_plot_model(model, out_img_path)
    if not ok:
        print("Falling back: saving model.summary() and a simple layer-params plot.")
        save_model_summary(model, summary_path)
        simple_layers_plot(model, layer_plot_path)
    else:
        # também salva summary textual
        save_model_summary(model, summary_path)

if __name__ == '__main__':
    main()
