"""Génère la model card (README.md) de l'auto-encodeur, à partir :
- des hyperparamètres et métriques d'entraînement déjà journalisés dans MLflow (perte, émissions CO2) ;
- des résultats d'évaluation déjà produits par `scripts/run_vision_pipeline.py`
  (`reports/figures/roc_curve.png`, `confusion_matrix.png`).

Aucun réentraînement ni recalcul d'inférence ici : le modèle est déjà entraîné et évalué, il ne s'agit
que de documenter l'existant.

    uv run python scripts/generate_model_card.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from src.vision.data import load_bin_gz
from src.vision.model_card import build_model_card, latest_finished_run

DATA_DIR = Path('deep-learning/screw')
MODELS_DIR = Path('models')

THRESHOLD_PERCENTILE = 95
# Lus sur reports/figures/roc_curve.png (légende AUROC) et confusion_matrix.png (titre = seuil,
# cellules = comptes), dernière évaluation produite par scripts/run_vision_pipeline.py.
ROC_AUC = 0.999
THRESHOLD = 0.030
CONFUSION_TN_FP_FN_TP = (41, 0, 2, 117)


def parse_args():
    parser = argparse.ArgumentParser(description="Génère la model card de l'auto-encodeur (README.md).")
    parser.add_argument('--model-name', default='autoencoder_conv_screw_ssim')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--experiment-name', default='b6_deep_learning_autoencodeur_16_16_16')
    parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
    parser.add_argument('--output', default=str(MODELS_DIR / 'README.md'))
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = latest_finished_run(client, args.experiment_name)
    print(f'Run MLflow : {run.info.run_id} ({args.experiment_name})')

    n_train = len(load_bin_gz(DATA_DIR / 'train_norm.bin.gz', args.img_size))

    card = build_model_card(
        run, roc_auc=ROC_AUC, threshold=THRESHOLD, threshold_percentile=THRESHOLD_PERCENTILE,
        confusion=CONFUSION_TN_FP_FN_TP, input_shape=(args.img_size, args.img_size, 1),
        n_train=n_train, model_name=args.model_name,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    card.save(output)
    print(f'Model card sauvegardée : {output}')


if __name__ == '__main__':
    main()
