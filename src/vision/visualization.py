"""Grilles d'images et histogrammes de score partagés entre les notebooks."""
import matplotlib.pyplot as plt
import numpy as np


def plot_score_histogram(scores, threshold, title, xlabel, threshold_label, color='tab:blue', precision=3):
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=20, color=color, alpha=0.8)
    plt.axvline(threshold, color='tab:red', linestyle='--',
                label=f'{threshold_label} = {threshold:.{precision}f}')
    plt.xlabel(xlabel)
    plt.ylabel("nombre d'images")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_image_grid(rows, suptitle, col_titles=None, row_labels=None, cmaps=None, figsize_per_cell=3,
                     save_path=None):
    """Grille d'images : `rows` est une liste de lignes, chaque ligne une liste d'images (une image par
    colonne). `col_titles` étiquette les colonnes (affiché sur la première ligne uniquement),
    `row_labels` étiquette chaque ligne (ylabel), `cmaps` fixe la colormap par colonne (gris par défaut).
    Si `save_path` est fourni, sauvegarde la figure à ce chemin (utile pour une pipeline non interactive).
    """
    n_rows, n_cols = len(rows), len(rows[0])
    cmaps = cmaps or ['gray'] * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, images in enumerate(rows):
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap=cmaps[col])
        if row_labels is not None:
            axes[row, 0].set_ylabel(row_labels[row], fontsize=9)

    if col_titles is not None:
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(suptitle)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
