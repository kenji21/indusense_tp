"""Pipeline DL bout en bout (MVTec screw) : charge les images prétraitées, construit l'auto-encodeur,
entraîne (MLflow + CodeCarbon), calcule score/seuil/AUROC, produit les heatmaps de défauts, puis
sauvegarde le modèle (models/) et les figures (reports/figures/).

    uv run python scripts/run_vision_pipeline.py --epochs 30 --img-size 128
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')

import numpy as np

from src.vision.autoencoder import build_autoencoder, reconstruction_score, train_autoencoder
from src.vision.data import build_defect_index, list_defect_types, load_bin_gz, load_mask
from src.vision.evaluation import plot_roc_and_confusion
from src.vision.visualization import plot_image_grid

DATA_DIR = Path('deep-learning/screw')
MODELS_DIR = Path('models')
FIGURES_DIR = Path('reports/figures')

PATIENCE = 4
MIN_DELTA = 1e-3
VALIDATION_SPLIT = 0.2
THRESHOLD_PERCENTILE = 95
N_HEATMAP_EXAMPLES = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraîne et évalue l'auto-encodeur de détection d'anomalies (MVTec screw)."
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--experiment-name', default='b6_deep_learning_autoencodeur_16_16_16')
    parser.add_argument('--run-name', default='autoencoder_conv_screw_ssim')
    return parser.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train_norm = load_bin_gz(DATA_DIR / 'train_norm.bin.gz', args.img_size)
    test_good_norm = load_bin_gz(DATA_DIR / 'test_good_norm.bin.gz', args.img_size)
    test_defects_norm = load_bin_gz(DATA_DIR / 'test_defects_norm.bin.gz', args.img_size)
    print(f'Train (normé)  : {X_train_norm.shape}')
    print(f'Test saines    : {test_good_norm.shape}')
    print(f'Test défauts   : {test_defects_norm.shape}')

    input_shape = X_train_norm.shape[1:]
    autoencoder, latent = build_autoencoder(input_shape, name='autoencodeur_screw')
    ratio = np.prod(latent.shape[1:]) / np.prod(input_shape)

    history = train_autoencoder(
        autoencoder, X_train_norm, latent.shape[1:], ratio,
        experiment_name=args.experiment_name, run_name=args.run_name,
        epochs=args.epochs, batch_size=args.batch_size, patience=PATIENCE, min_delta=MIN_DELTA,
        validation_split=VALIDATION_SPLIT,
    )
    print(f"Entraînement terminé : {len(history.history['loss'])} époque(s), "
          f"val_loss finale = {history.history['val_loss'][-1]:.5f}")

    split_at = int(len(X_train_norm) * (1 - VALIDATION_SPLIT))
    X_val_norm = X_train_norm[split_at:]
    errors_val = reconstruction_score(X_val_norm, autoencoder.predict(X_val_norm, verbose=0))
    threshold = np.percentile(errors_val, THRESHOLD_PERCENTILE)

    reconstructions_test_good = autoencoder.predict(test_good_norm, verbose=0)
    reconstructions_test_defects = autoencoder.predict(test_defects_norm, verbose=0)
    errors_test_good = reconstruction_score(test_good_norm, reconstructions_test_good)
    errors_test_defects = reconstruction_score(test_defects_norm, reconstructions_test_defects)

    y_true = np.concatenate([np.zeros(len(errors_test_good)), np.ones(len(errors_test_defects))])
    y_score = np.concatenate([errors_test_good, errors_test_defects])
    roc_auc, _ = plot_roc_and_confusion(
        y_true, y_score, threshold,
        threshold_label=f'seuil (centile {THRESHOLD_PERCENTILE}, validation)', save_dir=FIGURES_DIR,
    )
    print(f'AUROC (saines vs défauts, test) : {roc_auc:.4f}')

    test_dir = DATA_DIR / 'test'
    gt_dir = DATA_DIR / 'ground_truth'
    defect_types = list_defect_types(test_dir)
    _, mask_paths = build_defect_index(test_dir, gt_dir, defect_types)

    rows = []
    for idx in range(min(N_HEATMAP_EXAMPLES, len(test_defects_norm))):
        original = test_defects_norm[idx, ..., 0]
        reconstruction = reconstructions_test_defects[idx, ..., 0]
        heatmap = (original - reconstruction) ** 2
        mask = load_mask(mask_paths[idx], args.img_size)
        rows.append([original, reconstruction, heatmap, mask])
    plot_image_grid(
        rows, 'Défauts : original | reconstruction | heatmap | masque ground-truth',
        col_titles=['original', 'reconstruction', 'heatmap', 'masque ground-truth'],
        cmaps=['gray', 'gray', 'inferno', 'gray'],
        save_path=FIGURES_DIR / 'heatmaps.png',
    )

    model_path = MODELS_DIR / f'{args.run_name}.keras'
    autoencoder.save(model_path)
    print(f'Modèle sauvegardé : {model_path}')


if __name__ == '__main__':
    main()
