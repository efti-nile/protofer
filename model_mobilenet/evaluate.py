import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def predict_dataset(model, ds):

    _, labels_batched = tuple(zip(*ds))
    labels = tf.concat(labels_batched, axis=0)

    predictions_batched = model.predict(val_ds)
    predictions = tf.concat(predictions_batched, axis=0)

    labels = tf.argmax(labels, axis=1)
    predictions = tf.argmax(predictions, axis=1)

    return labels, predictions


def plot_confusion_matrix(labels, predictions):

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(labels)))
    plt.xticks(tick_marks, np.unique(labels))
    plt.yticks(tick_marks, np.unique(labels))

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()
