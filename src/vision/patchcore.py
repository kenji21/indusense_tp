"""PatchCore : détection d'anomalies par comparaison à une mémoire de patchs de features pré-entraînées."""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_feature_extractor(backbone_size, layer_name='conv3_block4_out'):
    """ResNet50 pré-entraîné sur ImageNet, gelé, tronqué à une couche intermédiaire."""
    backbone = keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(backbone_size, backbone_size, 3)
    )
    feature_extractor = keras.Model(backbone.input, backbone.get_layer(layer_name).output)
    feature_extractor.trainable = False
    return feature_extractor


def extract_patch_features(images, feature_extractor, backbone_size, batch_size=32):
    """images (N, H, W, 1) dans [0, 1] -> features patch (N, h, w, C). Nos images sont en niveaux de
    gris : on les convertit en RGB avant de les passer dans le backbone."""
    rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(images))
    rgb = tf.image.resize(rgb, (backbone_size, backbone_size))
    rgb = keras.applications.resnet50.preprocess_input(rgb * 255.0)
    return feature_extractor.predict(rgb, batch_size=batch_size, verbose=0)


def build_memory_bank(features, memory_size, seed):
    """Aplatit les features en patchs et sous-échantillonne à memory_size (mémoire complète si
    memory_size dépasse le nombre total de patchs)."""
    n, h, w, c = features.shape
    memory_bank_full = features.reshape(-1, c)

    rng = np.random.default_rng(seed)
    subset_idx = rng.choice(len(memory_bank_full), size=min(memory_size, len(memory_bank_full)), replace=False)
    return memory_bank_full[subset_idx], memory_bank_full.shape[0]


def anomaly_score(images, feature_extractor, nearest_neighbor, backbone_size):
    """Score par image = pire distance parmi ses patchs (un seul patch anormal suffit à signaler
    l'image, contrairement à une moyenne qui dilue le signal)."""
    features = extract_patch_features(images, feature_extractor, backbone_size)
    n, h, w, c = features.shape
    flat = features.reshape(-1, c)
    distances, _ = nearest_neighbor.kneighbors(flat)
    distance_maps = distances.reshape(n, h, w)
    return distance_maps.max(axis=(1, 2)), distance_maps
