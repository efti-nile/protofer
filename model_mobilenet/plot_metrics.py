import matplotlib.pyplot as plt


def plot_metrics(history):
    """
    This function takes the history from the model.fit() function and plots
    the training and validation loss and accuracy.

    Args:
    history: A dictionary containing the training and validation metrics.
    """

    # Create a figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the training and validation loss.
    if history['loss']:
        ax1.plot(history['loss'], label="Training Loss")
    if history['val_loss']:
        ax1.plot(history['val_loss'], label="Validation Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot the training and validation accuracy.
    if history['acc']:
        ax2.plot(history['acc'], label="Training Accuracy")
    if history['val_acc']:
        ax2.plot(history['val_acc'], label="Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    # Show the plot.
    plt.show()
