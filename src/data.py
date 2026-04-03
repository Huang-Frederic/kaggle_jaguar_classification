import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

from src.config import CLEANED_DIR, DATA_DIR, IMG_SIZE, BATCH_SIZE, SEED


def _load_images(df, img_dir):
    """Load images from a DataFrame with 'filename' column."""
    images = []
    for filename in df['filename']:
        img = Image.open(os.path.join(img_dir, filename)).convert('RGB')
        images.append(np.array(img, dtype=np.float32) / 255.0)
    return np.array(images)


def load_data(split='train'):
    """Load cleaned data from CSV + image folder.

    Parameters
    ----------
    split : 'train', 'val', or 'test'

    Returns
    -------
    For train/val : (images, labels)    — float32 array + int array
    For test      : (images, filenames) — float32 array + list of str
    """
    img_dir = os.path.join(CLEANED_DIR, 'train')

    if split == 'test':
        img_dir = os.path.join(CLEANED_DIR, 'test')
        fnames = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
        images = []
        for fname in fnames:
            img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
            images.append(np.array(img, dtype=np.float32) / 255.0)
        return np.array(images), fnames

    csv_path = os.path.join(CLEANED_DIR, f'{split}_split.csv')
    df = pd.read_csv(csv_path)
    images = _load_images(df, img_dir)
    labels = df['label'].values
    return images, labels


def load_test_pairs():
    """Load test pairs DataFrame."""
    return pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


def get_augmentation(level='none'):
    """Return a Keras augmentation layer.

    Levels
    ------
    none     : no augmentation (returns None)
    light    : horizontal flip only
    moderate : flip + brightness + contrast
    heavy    : flip + rotation + zoom + brightness + contrast
    """
    if level == 'none':
        return None

    if level == 'light':
        return keras.Sequential([
            layers.RandomFlip('horizontal'),
        ], name='aug_light')

    if level == 'moderate':
        return keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ], name='aug_moderate')

    if level == 'heavy':
        return keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.08),
            layers.RandomZoom((-0.1, 0.1)),
            layers.RandomBrightness(0.25),
            layers.RandomContrast(0.25),
        ], name='aug_heavy')

    raise ValueError(f"{level}")

def make_dataset(X, y, training=False, augmentation='none', batch_size=BATCH_SIZE):
    """Create a tf.data.Dataset with augmentation.

    Parameters
    ----------
    X            : numpy array of images (N, H, W, 3), float32 [0, 1]
    y            : numpy array of labels (N,)
    training     : shuffle the dataset
    augmentation : 'none', 'light', 'moderate', or 'heavy'
    batch_size   : batch size
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    ds = ds.batch(batch_size)

    aug = get_augmentation(augmentation)
    if training and aug is not None:
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)
