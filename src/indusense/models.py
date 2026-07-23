"""Aides à la construction des modèles de classification (maintenance prédictive)."""


def compute_scale_pos_weight(y_train):
    """Poids de la classe positive pour compenser le déséquilibre (utilisé par XGBoost)."""
    return (~y_train).sum() / y_train.sum()
