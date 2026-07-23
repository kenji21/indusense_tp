"""Évaluation des modèles de classification (maintenance prédictive)."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, pred):
    return {
        'pr_auc': average_precision_score(y_true, proba),
        'roc_auc': roc_auc_score(y_true, proba),
        'precision': precision_score(y_true, pred),
        'recall': recall_score(y_true, pred),
        'f1': f1_score(y_true, pred),
    }


def evaluate_models_cv(models, X_train_ts, y_train_ts, feature_cols, tscv):
    """PR-AUC moyen (et écart-type) par modèle sur les folds d'une TimeSeriesSplit. `models` doit
    contenir des modèles déjà fit (clonés — donc réinitialisés — avant chaque fold)."""
    cv_scores = {name: [] for name in models}

    for train_idx, val_idx in tscv.split(X_train_ts):
        X_fold_train = X_train_ts.iloc[train_idx][feature_cols]
        y_fold_train = y_train_ts.iloc[train_idx]
        X_fold_val = X_train_ts.iloc[val_idx][feature_cols]
        y_fold_val = y_train_ts.iloc[val_idx]

        for name, fitted_model in models.items():
            model = clone(fitted_model)
            model.fit(X_fold_train, y_fold_train)
            proba = model.predict_proba(X_fold_val)[:, 1]
            cv_scores[name].append(average_precision_score(y_fold_val, proba))

    return pd.DataFrame({
        'Modèle': cv_scores.keys(),
        'PR-AUC moyen (CV)': [pd.Series(s).mean() for s in cv_scores.values()],
        'Écart-type (CV)': [pd.Series(s).std() for s in cv_scores.values()],
    }).sort_values('PR-AUC moyen (CV)', ascending=False).reset_index(drop=True)


def plot_confusion_matrices(models, X, y):
    """Table + matrices de confusion (avec libellés de quadrant) pour chaque modèle."""
    predictions = {name: model.predict(X) for name, model in models.items()}

    cm_table = pd.DataFrame([
        {'Modèle': name, 'Vrai Négatif': tn, 'Faux Positif': fp, 'Faux Négatif': fn, 'Vrai Positif': tp}
        for name, pred in predictions.items()
        for tn, fp, fn, tp in [confusion_matrix(y, pred).ravel()]
    ])
    print(cm_table.to_string(index=False))

    quadrant_labels = [['Vrai Négatif', 'Faux Positif'], ['Faux Négatif', 'Vrai Positif']]

    fig, axes = plt.subplots(1, len(models), figsize=(15, 4))
    for ax, (name, pred) in zip(axes, predictions.items()):
        disp = ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False)
        for i in range(2):
            for j in range(2):
                value = disp.text_[i, j].get_text()
                disp.text_[i, j].set_text(f'{value}\n{quadrant_labels[i][j]}')
        ax.set_title(name)

    plt.tight_layout()
    plt.show()
