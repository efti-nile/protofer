import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


AUTOTUNE = tf.data.AUTOTUNE

def get_dataset(
    ds_dir,
    img_width,
    img_height,
    batch_size,
    augment=None,
    color_mode='grayscale',
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
        shuffle_buffer_size: Buffer size for shuffle.
        seed:
    Returns:
        A tf.data.Dataset objects.
    """

    ds = image_dataset_from_directory(
        ds_dir,
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=color_mode
    )

    class_names = ds.class_names
    print("Class Names:", class_names)

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
