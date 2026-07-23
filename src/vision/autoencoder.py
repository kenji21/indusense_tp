"""Auto-encodeur convolutionnel pour la détection d'anomalies par erreur de reconstruction."""
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_shape, name='autoencodeur_screw'):
    """Encodeur : 3 convolutions stridées, filtres resserrés (32 -> 64 -> 16) pour forcer un vrai goulot.
    Décodeur : noyau=4 (multiple du stride=2) plutôt que 3, pour éviter l'artefact de damier
    ("checkerboard") typique des Conv2DTranspose quand noyau et stride ne s'accordent pas.
    """
    inputs = keras.Input(shape=input_shape, name='image')

    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    latent = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(latent)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')(x)

    autoencoder = keras.Model(inputs, outputs, name=name)
    return autoencoder, latent


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def train_autoencoder(autoencoder, X_train, latent_shape, ratio, experiment_name, run_name,
                       epochs=80, batch_size=16, patience=4, min_delta=1e-3,
                       validation_split=0.2, tracking_uri='sqlite:///mlflow.db'):
    """Entraîne l'auto-encodeur (perte SSIM) avec early stopping, et journalise l'entraînement dans MLflow.

    Arrête l'entraînement si val_loss ne s'améliore plus d'au moins min_delta pendant patience époques.
    Sans min_delta, un nouveau record de +0.0001 compte comme une "amélioration" et remet le compteur à
    zéro indéfiniment. restore_best_weights ramène le modèle à l'époque où val_loss était la meilleure,
    pas la dernière.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    autoencoder.compile(optimizer='adam', loss=ssim_loss)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'optimizer': 'adam',
            'loss': 'ssim',
            'decoder_kernel_size': 4,
            'early_stopping_patience': patience,
            'early_stopping_min_delta': min_delta,
            'latent_shape': str(tuple(latent_shape)),
            'compression_ratio': round(float(ratio), 4),
        })

        history = autoencoder.fit(
            X_train, X_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=2,
        )

        mlflow.log_param('stopped_epoch', early_stopping.stopped_epoch or len(history.history['loss']))
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
            mlflow.log_metrics({'loss': loss, 'val_loss': val_loss}, step=epoch)

    return history


def reconstruction_score(images, reconstructions, percentile=99):
    """Score par image = centile `percentile` de l'erreur pixel (et non la moyenne).

    Un défaut n'occupe qu'une petite zone de l'image : sa contribution à une moyenne sur toute l'image
    est diluée par le bruit de reconstruction du filetage (présent même sur des saines). Le centile 99
    capte le pic d'erreur local, plus discriminant qu'une moyenne globale.
    """
    squared_error = (images - reconstructions) ** 2
    return np.percentile(squared_error, percentile, axis=(1, 2, 3))
