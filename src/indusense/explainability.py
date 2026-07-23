"""Chargement de modèles depuis MLflow et construction d'explications SHAP."""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_best_run(experiment_name, filter_string):
    """Le run avec la meilleure PR-AUC (validation) correspondant au filtre, dans l'expérience donnée."""
    return (
        mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string)
        .sort_values('metrics.pr_auc', ascending=False)
        .iloc[0]
    )


def load_run_model(run):
    model_uri = f'runs:/{run.run_id}/model'
    try:
        return mlflow.sklearn.load_model(model_uri)
    except mlflow.exceptions.MlflowException:
        return mlflow.xgboost.load_model(model_uri)


def load_best_run_model(experiment_name, model_name_pattern):
    """Charge le modèle du meilleur run (par PR-AUC) qui se charge effectivement, en sautant les runs
    éventuellement mal loggés (mauvais flavor, artefact absent)."""
    candidates = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.model_name LIKE '{model_name_pattern}'",
    ).sort_values('metrics.pr_auc', ascending=False)

    for _, run in candidates.iterrows():
        model_uri = f'runs:/{run.run_id}/model'
        for loader in (mlflow.sklearn.load_model, mlflow.xgboost.load_model):
            try:
                model = loader(model_uri)
            except mlflow.exceptions.MlflowException:
                continue
            if hasattr(model, 'predict_proba'):  # écarte les runs où un objet non-modèle a été loggé
                return run, model
        print(f"Run ignoré (modèle illisible) : {run['tags.model_name']!r} ({run.run_id})")

    raise RuntimeError(f'Aucun run chargeable pour le motif {model_name_pattern!r}')


def transform_for_shap(model, X):
    """Aligne X sur ce que l'explainer attend : données standardisées pour un Pipeline (ex. régression
    logistique), inchangées pour un modèle à base d'arbres (Random Forest / XGBoost)."""
    if isinstance(model, Pipeline):
        return pd.DataFrame(model[:-1].transform(X), columns=X.columns, index=X.index)
    return X


def build_shap_explainer(model, X_background):
    """Pipeline (régression logistique standardisée) -> LinearExplainer sur les données transformées.
    Modèle à base d'arbres -> TreeExplainer. Retourne (explainer, X transformé)."""
    X_expl = transform_for_shap(model, X_background)
    if isinstance(model, Pipeline):
        explainer = shap.LinearExplainer(model[-1], X_expl)
    else:
        explainer = shap.TreeExplainer(model)
    return explainer, X_expl


def shap_failure_class(shap_values):
    """Pour un modèle multi-classe (Random Forest : une sortie par classe), ne garde que la classe
    "panne" (indice 1). Sans effet pour un modèle à sortie binaire native (XGBoost, régression
    logistique)."""
    return shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values
