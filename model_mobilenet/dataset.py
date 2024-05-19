import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory


AUTOTUNE = tf.data.AUTOTUNE

def get_dataset(
    ds_dir,
    img_width,
    img_height,
    batch_size,
    augment=None,
    color_mode='grayscale',
    convert_to_rgb=False,
    shuffle_buffer_size=5000,
    seed=123
):

    """
    Returns a dataset object created from an image folder.
    
    Args:
        ds_dir: A path to images. The path has to contain N folders with images,
            folders named as class names.
        img_width:
        img_height:
        batch_size (int): Split pairs into batches of this size.
        augment (tf.keras.Sequential): Augmentations to be aplied. Should be
            specified as a sequential keras model which contains all necessary
            transformations as layers.
        color_mode:
        convert_to_rgb: convert read grayscale images to RGB
        shuffle_buffer_size: Buffer size for shuffle.
        seed:
    Returns:
        A tf.data.Dataset objects.
    """
    
    if convert_to_rgb and color_mode != 'grayscale':
        ValueError("convert_to_rgb flag is only for grayscale mode")

    ds = image_dataset_from_directory(
        ds_dir,
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=color_mode
    )

    class_names = ds.class_names
    print("Class Names:", class_names)
    
    if convert_to_rgb:
        
        def convert(image, label):
            image = tf.image.grayscale_to_rgb(image)
            return image, label
        
        ds = ds.map(convert)

    def one_hot_encode(labels):
        num_classes = len(class_names)
        return tf.one_hot(labels, num_classes)

    ds = ds.map(lambda image, label: (image, one_hot_encode(label)))

    if augment:
        ds = ds.map(lambda image, label: (augment(image), label))

    def normalize(ds):
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
        return normalized_ds

    ds = normalize(ds)

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    ds = configure_for_performance(ds)

    return ds


def save_images(ds, classes, save_dir) -> None:
    """
    Save images from a dataset to a directory.
    Args:
        ds (tf.data.Dataset): Dataset
        classes (list[str]): List of class names
        save_dir: Directory to save images
    """
    for i, batch in enumerate(ds):
        images, targets = batch
        for j, image in enumerate(images):
            class_name = classes[np.argmax(targets[j])]
            image_path = os.path.join(save_dir, f"{i:03}_{j:03}_{class_name}.png")
            tf.keras.preprocessing.image.save_img(image_path, image)


def print_counts(ds, classes) -> None:
    """
    Print the number of images for each class.
    Args:
        ds (tf.data.Dataset): Dataset
        classes (list[str]): List of class names
    """
    counts = {c: 0 for c in classes}
    total = 0
    for batch in ds:
        _, targets = batch
        for t in targets:
            class_name = classes[np.argmax(t)]
            counts[class_name] += 1
            total += 1
    for c in classes:
        print(f"{c}: {counts[c]}")
    print(f"Total: {total}")


def visualize(ds):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")


def print_shapes(ds):
    for images, labels in ds.take(1):
        print(images.shape)
        print(labels.shape)
