"""Construction de la model card (Hugging Face) de l'auto-encodeur, à partir d'un run MLflow et
d'une évaluation (AUROC, matrice de confusion) sur le jeu de test MVTec (screw)."""
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.repocard_data import EvalResult

DATASET_URL = 'https://www.mvtec.com/company/research/datasets/mvtec-ad'


def latest_finished_run(client, experiment_name):
    """Run MLflow le plus récent (statut FINISHED) d'une expérience donnée."""
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=['start_time DESC'],
        max_results=1,
    )
    return runs[0]


def build_model_card(run, roc_auc, threshold, threshold_percentile, confusion,
                      input_shape, n_train, model_name):
    """Construit la model card HF à partir des hyperparamètres/métriques d'un run MLflow terminé et
    des résultats d'évaluation (recalculés sur le jeu de test au moment de la génération)."""
    params = run.data.params
    metrics = run.data.metrics
    tn, fp, fn, tp = confusion
    height, width = input_shape[0], input_shape[1]

    card_data = ModelCardData(
        language='fr',
        license='mit',
        library_name='keras',
        tags=['anomaly-detection', 'autoencoder', 'computer-vision', 'mvtec-ad', 'unsupervised-learning'],
        datasets=['mvtec-ad'],
        eval_results=[
            EvalResult(
                task_type='anomaly-detection',
                dataset_type='mvtec-ad',
                dataset_name='MVTec AD (screw)',
                metric_type='auroc',
                metric_value=round(roc_auc, 4),
                metric_name='AUROC (image, saines vs défauts)',
            ),
        ],
        model_name=model_name,
    )

    stopped_epoch = params.get('stopped_epoch', params.get('epochs'))
    co2_emitted = None
    if 'emissions_kg_co2eq' in metrics:
        co2_emitted = (
            f"{metrics['emissions_kg_co2eq'] * 1000:.3f} g CO₂eq "
            f"({metrics.get('energy_consumed_kwh', 0):.5f} kWh, mesuré avec CodeCarbon, France)"
        )

    card = ModelCard.from_template(
        card_data,
        model_id=model_name,
        model_summary=(
            "Auto-encodeur convolutionnel non supervisé pour la détection d'anomalies visuelles sur des vis "
            "(catégorie `screw` du jeu [MVTec AD](" + DATASET_URL + ")), entraîné uniquement sur des pièces "
            "saines : tout ce qu'il reconstruit mal devient suspect."
        ),
        model_description=(
            f"Encodeur à 3 convolutions stridées (32→64→16 filtres) compressant l'image "
            f"{height}×{width} en niveaux de gris vers un goulot {params.get('latent_shape')} "
            f"(ratio de compression {params.get('compression_ratio')}). Décodeur symétrique à 3 "
            f"déconvolutions (noyau {params.get('decoder_kernel_size', 4)}, multiple du stride 2 pour "
            "éviter l'artefact de damier), sortie sigmoïde. Le score d'anomalie d'une image est le "
            "centile 99 de l'erreur de reconstruction par pixel ; le seuil de décision est calibré sur le "
            f"centile {threshold_percentile} des erreurs mesurées sur les pièces saines de validation."
        ),
        model_type='auto-encodeur convolutionnel (CNN)',
        developers='Équipe IndSense',
        training_data=(
            f"{n_train} images saines de la catégorie `screw` du jeu [MVTec AD]({DATASET_URL}), "
            f"normalisées ([0, 1]) et redimensionnées à {height}×{width}."
        ),
        preprocessing=f'Niveaux de gris, redimensionnement {height}×{width}, normalisation min-max [0, 1].',
        training_regime=(
            f"Optimiseur {params.get('optimizer')}, perte {params.get('loss')} (1 - SSIM), "
            f"batch size {params.get('batch_size')}, {params.get('epochs')} époques max "
            f"(arrêt à l'époque {stopped_epoch} par early stopping, patience="
            f"{params.get('early_stopping_patience')}, min_delta={params.get('early_stopping_min_delta')})."
        ),
        speeds_sizes_times=(
            f"Perte finale (train / validation) : {metrics.get('loss', float('nan')):.4f} / "
            f"{metrics.get('val_loss', float('nan')):.4f} (perte SSIM)."
        ),
        testing_data=(
            f"Jeu de test MVTec AD `screw` ({tn + fp} pièces saines, {tp + fn} défauts réels : "
            "`manipulated_front`, `scratch_head`, `scratch_neck`, `thread_side`, ...), avec masques de "
            "vérité terrain pixel par pixel."
        ),
        testing_metrics=(
            f"AUROC au niveau image (saines vs défauts) et matrice de confusion au seuil = centile "
            f"{threshold_percentile} de l'erreur de reconstruction sur les saines de validation."
        ),
        results=(
            f"- **AUROC (image)** : {roc_auc:.3f}\n"
            f"- **Seuil de décision** : {threshold:.4f} (centile {threshold_percentile}, saines de validation)\n"
            f"- **Matrice de confusion (test)** : VN={tn}, FP={fp}, FN={fn}, VP={tp}\n"
            f"- **Rappel défauts** : {tp / (tp + fn):.3f} — **Taux de faux positifs (saines)** : "
            f"{fp / (tn + fp):.3f}"
        ),
        model_examination=(
            "Voir `reports/figures/heatmaps.png` : original | reconstruction | heatmap d'erreur par pixel | "
            "masque ground-truth, sur des exemples de défauts réels."
        ),
        bias_risks_limitations=(
            "Entraîné exclusivement sur des pièces saines de la catégorie `screw` : ne généralise pas aux "
            "types de défauts absents du jeu de test, ni à d'autres catégories d'objets ou de textures. Un "
            "goulot d'étranglement trop large risquerait d'apprendre une quasi-identité et de reconstruire "
            "aussi bien les défauts que les pièces saines, faisant disparaître le signal d'anomalie "
            f"(ratio de compression actuel : {params.get('compression_ratio')})."
        ),
        hardware_type='CPU (Apple M3)',
        cloud_provider='local',
        cloud_region='France',
        co2_emitted=co2_emitted,
        model_specs=(
            f"Auto-encodeur convolutionnel, entrée {height}×{width}×1, goulot "
            f"{params.get('latent_shape')}, perte SSIM."
        ),
        software='TensorFlow/Keras, MLflow (suivi des runs), CodeCarbon (empreinte carbone).',
        model_card_authors='Équipe IndSense',
    )
    return card
