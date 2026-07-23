# InduSense TP — B5 Maintenance prédictive

Voir [b5_maintenance_ml.md](b5_maintenance_ml.md) pour l'énoncé complet du TP.

## Prérequis

- Le dépôt [`indusense`](../indusense) cloné en tant que dossier frère (`../indusense`), contenant le Gold dataset au format parquet (`gold_dataset_*.parquet`).
- [uv](https://docs.astral.sh/uv/) installé.

## Installation

```bash
uv sync
```

Installe les dépendances du projet (`pandas`, `pyarrow`) ainsi que les outils de dev (`jupyter`, `ipykernel`).

## Notebooks

| Notebook | Contenu |
|---|---|
| [`01_maintenance_ml.ipynb`](01_maintenance_ml.ipynb) | Chargement du Gold dataset (`../indusense/gold_dataset_20260622-080603.parquet`) et premier aperçu (shape, `head`, `info`). |

### Lancer un notebook

```bash
uv run jupyter lab
```

### Exécuter un notebook en ligne de commande (non interactif)

```bash
uv run jupyter nbconvert --to notebook --execute --inplace 01_maintenance_ml.ipynb
```

## Suivi des expériences avec MLflow

Les entraînements sont journalisés dans une base SQLite locale (`mlflow.db`). Pour ouvrir l'UI MLflow :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir [http://localhost:5000](http://localhost:5000) dans le navigateur.
