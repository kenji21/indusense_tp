"""Chargement, sauvegarde et indexation des images du dataset MVTec AD (screw)."""
import gzip
from pathlib import Path

import cv2
import numpy as np


def list_defect_types(test_dir):
    """Catégories de défauts = sous-dossiers de test_dir, hors 'good', triés."""
    return sorted(p.name for p in Path(test_dir).iterdir() if p.is_dir() and p.name != 'good')


def load_image(path, size, normalize=False):
    """Charge en niveaux de gris et redimensionne. uint8 [0, 255] par défaut, float32 [0, 1] si normalize."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img[..., np.newaxis]
    return img.astype('float32') / 255.0 if normalize else img


def load_folder(folder, size, normalize=False):
    """Charge toutes les images .png d'un dossier (ordre trié)."""
    paths = sorted(Path(folder).glob('*.png'))
    images = np.stack([load_image(p, size, normalize=normalize) for p in paths])
    return images, paths


def load_test_defects_by_category(test_dir, defect_types, size, normalize=False):
    """dict {défaut: (images, paths)} — un chargement par catégorie de défaut."""
    return {defect: load_folder(Path(test_dir) / defect, size, normalize=normalize) for defect in defect_types}


def build_defect_index(test_dir, gt_dir, defect_types):
    """Catégorie et chemin du masque ground-truth pour chaque image de test défectueuse, dans l'ordre
    (catégories triées, puis fichiers triés dans chaque catégorie) — même ordre qu'une concaténation
    par catégorie des images de test."""
    category_per_index = []
    mask_paths = []
    for defect in defect_types:
        img_paths = sorted((Path(test_dir) / defect).glob('*.png'))
        category_per_index += [defect] * len(img_paths)
        mask_paths += [Path(gt_dir) / defect / f'{p.stem}_mask.png' for p in img_paths]
    return category_per_index, mask_paths


def load_mask(path, size):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def normalize(images):
    return images.astype('float32') / 255.0


def save_bin(array, path):
    with gzip.open(path, 'wb') as f:
        f.write(array.tobytes())
    print(f'{path} : {array.shape} {array.dtype} -> {path.stat().st_size / 1e6:.1f} Mo (gzip)')


def load_bin_gz(path, img_size):
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='float32')
    return data.reshape(-1, img_size, img_size, 1)
