"""Évaluation partagée entre les approches (ROC, matrice de confusion, analyse des faux négatifs)."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


def plot_roc_and_confusion(y_true, y_score, threshold, threshold_label, precision=3, save_dir=None):
    """Courbe ROC (avec AUROC) puis matrice de confusion au seuil donné. Retourne (roc_auc, y_pred).

    Si `save_dir` est fourni, sauvegarde `roc_curve.png` et `confusion_matrix.png` dans ce dossier
    (utile pour une pipeline non interactive)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('taux de faux positifs')
    plt.ylabel('taux de vrais positifs')
    plt.title('Courbe ROC (saines vs défauts, test)')
    plt.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(Path(save_dir) / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    y_pred = (y_score > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    ConfusionMatrixDisplay(cm, display_labels=['saine', 'défaut']).plot(cmap='Blues')
    plt.title(f'Matrice de confusion ({threshold_label} = {threshold:.{precision}f})')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    return roc_auc, y_pred


def missed_defect_counts(y_true, y_pred, category_per_index, n_good):
    """Faux négatifs (défaut prédit saine) par catégorie de défaut. category_per_index doit être aligné
    sur l'ordre de concaténation des images de test défectueuses (cf. build_defect_index)."""
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0] - n_good
    missed = [category_per_index[i] for i in fn_idx]
    return Counter(missed), fn_idx


def plot_missed_defects(counts, defect_types):
    for defect in defect_types:
        print(f'{defect:<20}: {counts.get(defect, 0)} loupé(s)')

    plt.figure(figsize=(6, 4))
    plt.bar(defect_types, [counts.get(d, 0) for d in defect_types], color='tab:red')
    plt.ylabel('nombre de faux négatifs')
    plt.title('Origine des défauts prédits saine (faux négatifs)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
