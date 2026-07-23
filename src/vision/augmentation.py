"""Pipeline d'augmentation appliqué aux saines d'entraînement (auto-encodeur)."""
import albumentations as A
import cv2


def build_augmentation_pipeline():
    """Transformations légères : flips, rotation <= 15°, translation/échelle <= 5%, luminosité/contraste.

    Pas de transformations "elastic" ou "grid" (rendraient une vis saine artificiellement défectueuse) ni
    d'occlusion (masquerait un éventuel défaut plutôt que de le simuler).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT101, p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])


def augment_image(img, pipeline):
    """img est déjà brut (uint8) — Albumentations travaille nativement dans [0, 255]."""
    return pipeline(image=img)['image']
