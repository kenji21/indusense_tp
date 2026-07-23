# Rapport — Auto-encodeur & détection d'anomalies (MVTec AD `screw`)

Résumé du cheminement du TP `tp_deep_learning.ipynb` (partie 2), catégorie **screw**. Historique détaillé : `git log -- deep-learning/`.

## Partie 1 — Données (rappel)

- Chargement + redimensionnement (128×128) en **images brutes** (`uint8`), transformations (augmentation Albumentations : flips, rotation ≤15°, translation/échelle ≤5%) appliquées avant la **normalisation finale** dans `[0, 1]`.
- Jeux exportés en `.bin.gz` (train saines, test saines, test défauts) pour découpler le prétraitement de l'entraînement du modèle.

## Partie 2 — Modèle

### Étape 1 — Architecture

Auto-encodeur convolutionnel : 3 convolutions (encodeur, 32→64→8 filtres) + 3 déconvolutions (décodeur), sortie `sigmoid`.
- Noyau **4** (et non 3) dans le décodeur : évite l'artefact de damier (*checkerboard*) propre aux `Conv2DTranspose` quand noyau et stride ne s'accordent pas.
- Goulot initial **16×16×8** (2048 valeurs, ratio 0.125, compression ×8) — volontairement resserré pour éviter que le modèle apprenne une quasi-identité (auquel cas il reconstruirait aussi bien les défauts que les saines).

### Étape 2 — Entraînement

- Perte **SSIM** (`1 - tf.image.ssim`) plutôt que MSE — plus sensible aux altérations de texture.
- `EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-3)` : sans `min_delta`, `val_loss` battait sans cesse des micro-records (+0.0001) et l'entraînement allait jusqu'à la limite d'époques sans jamais s'arrêter.
- Suivi MLflow (expérience `b6_deep_learning_autoencodeur_2`).

### Étape 3 — Score d'anomalie & seuil

- Score initial = **MSE moyenne par image**. Problème : un défaut n'occupe qu'une petite zone, sa contribution à une moyenne globale est diluée par le bruit de reconstruction du filetage (présent même sur des saines).
- Score corrigé = **centile 99 de l'erreur pixel** par image (capte le pic local plutôt que la moyenne) → AUROC 0.978.
- Seuil calibré sur `test_good_norm` (données jamais vues, ni en entraînement ni en sélection de modèle) plutôt que sur le split de validation interne du `fit()`, qui avait influencé le choix des poids via `restore_best_weights`.

### Étape 4 — Heatmaps & évaluation

- Visualisation `original | reconstruction | heatmap | masque ground-truth` sur des défauts, puis sur les faux positifs/négatifs spécifiquement.
- Matrice de confusion (goulot 8 canaux, seuil centile 95) : 39 saines correctes / 2 faux positifs, 99 défauts corrects / 20 faux négatifs.
- Répartition des faux négatifs par catégorie : `manipulated_front` le plus loupé (défaut le plus "global", pas une simple texture locale).

## Grosse évolution : goulot 16×16×8 → 16×16×16

**Diagnostic** : sur les faux positifs/négatifs affichés, la reconstruction restait floue et générique même sur des saines — le goulot à 8 canaux sous-apprenait la texture normale du filetage. Ce bruit de fond, présent partout, noyait le signal des petits défauts plutôt que de simplement les rater faute de compression suffisante.

**Action** : goulot élargi à **16 canaux** (16×16×16, ratio 0.25, compression ×4, 77 809 paramètres) pour permettre une reconstruction plus fidèle de la texture normale, sans perdre la compression nécessaire à la détection d'anomalie.

**Résultat** : AUROC 0.978 → **0.982**, faux négatifs 20 → **16** (à seuil constant). Gain réel mais modeste — cohérent avec l'hypothèse (le goulot était bien un facteur limitant) sans être une rupture radicale.

## Pour aller plus loin

- **`manipulated_front`** reste le défaut le plus difficile à détecter — probablement parce qu'il altère une zone plus globale de la tête de vis plutôt qu'une texture locale.
- Piste non testée : réduire la résolution spatiale du goulot de 16×16 à **8×8**, en compensant par plus de canaux (ex. 8×8×64 = 4096 valeurs, capacité totale égale à la version actuelle 16×16×16). Chaque cellule du goulot résumerait alors une zone 2× plus grande de l'image, forçant un encodage plus global/structurel — potentiellement pertinent pour un défaut comme `manipulated_front`, au risque de pénaliser les défauts fins et localisés (`scratch_head`, `scratch_neck`, filetage). À valider empiriquement.
