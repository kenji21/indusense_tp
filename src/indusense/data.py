"""Chargement et préparation du Gold dataset (maintenance prédictive)."""
import pandas as pd


def load_gold_dataset(path):
    gold = pd.read_parquet(path)
    return gold.sort_values(['machine_id_std', 'window_start']).reset_index(drop=True)


def drop_leaking_columns(gold):
    """Retire les colonnes qui fuient l'information du futur (`future_*`, `*_next_*`) ainsi que les
    bornes de fenêtre temporelle (`window_start`/`window_end`, non pertinentes comme features)."""
    dropped_cols = [c for c in gold.columns if c.startswith('future_') or '_next_' in c]
    dropped_cols += ['window_start', 'window_end']
    gold = gold.drop(columns=dropped_cols)

    print(f'Colonnes supprimées ({len(dropped_cols)}) :', dropped_cols)
    print('Shape :', gold.shape)
    return gold, dropped_cols


def temporal_split(gold, y, split_col='split_set'):
    """Découpe train/validation/test à partir d'une colonne de split déjà temporelle (train < validation
    < test dans le temps)."""
    X = gold.drop(columns=split_col)

    splits = {}
    for name in ('train', 'validation', 'test'):
        mask = gold[split_col] == name
        splits[name] = (X.loc[mask], y.loc[mask])
        print(f'{name:<12} {mask.sum():>7} lignes')
    return splits


def impute_missing(X, X_train, X_val, X_test, zero_cols):
    """Médiane (calculée sur X_train, pour éviter la fuite) pour les colonnes à NaN, sauf `zero_cols`
    (absence d'incident = 0, pas une valeur manquante à imputer par la médiane). Les colonnes à imputer
    sont déterminées sur `X` (train+validation+test), pour couvrir un NaN qui n'apparaîtrait que dans un
    split donné. Mutation en place de X_train/X_val/X_test, comme le fillna par colonne d'origine."""
    nan_counts = X.isna().sum()
    median_cols = [c for c in nan_counts[nan_counts > 0].index if c not in zero_cols]

    medians = X_train[median_cols].median()

    for split in (X_train, X_val, X_test):
        split[median_cols] = split[median_cols].fillna(medians)
        split[zero_cols] = split[zero_cols].fillna(0)

    print('NaN restants — train :', X_train.isna().sum().sum())
    print('NaN restants — val   :', X_val.isna().sum().sum())
    print('NaN restants — test  :', X_test.isna().sum().sum())


def build_chronological_order(gold_file, index):
    """Recalcule `window_start` (retiré des features par drop_leaking_columns), aligné sur le même tri
    et la même ligne supprimée que la préparation des features, pour ordonner chronologiquement un
    sous-ensemble d'index (ex. X_train) avant une validation croisée temporelle."""
    window_start_full = (
        pd.read_parquet(gold_file)
        .sort_values(['machine_id_std', 'window_start'])['window_start']
        .reset_index(drop=True)
        .iloc[1:]
        .reset_index(drop=True)
    )
    return window_start_full.loc[index].sort_values().index
