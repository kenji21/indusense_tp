"""Suivi MLflow partagé pour les expériences de maintenance prédictive."""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier

from src.indusense.evaluation import classification_metrics


def log_model_flavor(model):
    """xgboost.log_model pour un XGBClassifier natif, sklearn.log_model sinon (y compris les Pipeline
    scikit-learn, ex. régression logistique standardisée)."""
    logger = mlflow.xgboost if isinstance(model, XGBClassifier) else mlflow.sklearn
    logger.log_model(model, name='model')


def log_classification_run(stage, model_name, params, model, X_eval, y_eval, extra_metrics=None):
    """À appeler à l'intérieur d'un `with mlflow.start_run(...)` : tags, hyperparamètres, métriques de
    classification sur (X_eval, y_eval) et le modèle lui-même (flavor détecté automatiquement)."""
    mlflow.set_tag('stage', stage)
    mlflow.set_tag('model_name', model_name)
    mlflow.log_params(params)

    proba = model.predict_proba(X_eval)[:, 1]
    pred = model.predict(X_eval)
    metrics = classification_metrics(y_eval, proba, pred)
    if extra_metrics:
        metrics.update(extra_metrics)
    mlflow.log_metrics(metrics)

    log_model_flavor(model)


def log_cv_child_runs(cv_results, scoring, run_name_prefix, strip_prefix=None):
    """À appeler à l'intérieur du `with mlflow.start_run(...)` parent : un run enfant (nested) par
    combinaison de la grille, avec ses scores CV moyens (par fold)."""
    def clean(params):
        if strip_prefix is None:
            return params
        return {k.removeprefix(strip_prefix): v for k, v in params.items()}

    for i, params in enumerate(cv_results['params']):
        combo = ', '.join(f'{k}={v}' for k, v in clean(params).items())
        with mlflow.start_run(run_name=f'{run_name_prefix} ({combo})', nested=True):
            mlflow.set_tag('stage', 'gridsearch')
            mlflow.log_params(clean(params))
            mlflow.log_metrics({
                **{f'cv_{name}': cv_results[f'mean_test_{name}'][i] for name in scoring},
                'cv_pr_auc_std': cv_results['std_test_pr_auc'][i],
                'cv_rank': cv_results['rank_test_pr_auc'][i],
            })

    print(f'{len(cv_results["params"])} runs enfants loggés (1 par combinaison)')


def log_optuna_child_runs(study, run_name_prefix):
    """À appeler à l'intérieur du `with mlflow.start_run(...)` parent : un run enfant (nested) par essai
    Optuna, avec son score CV."""
    for trial in study.trials:
        combo = ', '.join(f'{k}={v}' for k, v in trial.params.items())
        with mlflow.start_run(run_name=f'{run_name_prefix} trial {trial.number} ({combo})', nested=True):
            mlflow.set_tag('stage', 'optuna')
            mlflow.log_params(trial.params)
            mlflow.log_metrics({'cv_pr_auc': trial.value, 'trial_number': trial.number})

    print(f'{len(study.trials)} runs enfants loggés (1 par essai)')


def format_run_comparison(runs):
    """Tableau de comparaison des runs MLflow (le plus récent par modèle), avec un résumé texte des
    hyperparamètres pour affichage."""
    param_cols = [c for c in runs.columns if c.startswith('params.')]
    runs = runs.copy()
    runs['Hyper paramètres'] = runs[param_cols].apply(
        lambda r: '\n'.join(f"{c.removeprefix('params.')}={v}" for c, v in r.items() if pd.notna(v)),
        axis=1,
    )

    return (
        runs.rename(columns={
            'tags.model_name': 'Modèle',
            'metrics.pr_auc': 'PR-AUC',
            'metrics.roc_auc': 'ROC-AUC',
            'metrics.precision': 'Précision',
            'metrics.recall': 'Rappel',
            'metrics.f1': 'F1',
        })
        .sort_values('start_time', ascending=False)
        .drop_duplicates('Modèle')
        [['Modèle', 'Hyper paramètres', 'PR-AUC', 'ROC-AUC', 'Précision', 'Rappel', 'F1']]
        .sort_values('PR-AUC', ascending=False)
        .reset_index(drop=True)
    )
