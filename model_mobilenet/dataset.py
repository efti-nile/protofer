import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


AUTOTUNE = tf.data.AUTOTUNE

def get_dataset(
    ds_dir,
    batch_size=64,
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
        image_size=(img_size, img_size),
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
        ds = ds.map(lambda image, label: (data_augmentation(image), label))

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

