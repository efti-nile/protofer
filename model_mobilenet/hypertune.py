import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from train import train


def grid_search(model, train_ds, val_ds, lr_list, bs_list, epochs, loss, save=None) -> tuple[float, float]:

    """
    Grid search for the best learning rate and batch size.
    Args:
        model (keras.model): Model
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        lr_list (list[float]): List of learning rates
        bs_list (list[int]): List of batch sizes
        save (Path, str, None): Save search results to this folder
    Returns:
        lr_best (float): Best learning rate
        bs_best (float): Best batch size
    """

    val_losses = {}

    weights = model.get_weights()

    for learning_rate in lr_list:

        for batch_size in bs_list:

            print(f"Using learning rate: {learning_rate} and batch size: {batch_size}")

            train_ds = train_ds.rebatch(batch_size)
            val_ds = val_ds.rebatch(batch_size)

            model.set_weights(weights)

            result = train(
                model,
                train_ds,
                val_ds,
                learning_rate,
                loss,
                epochs,
                batch_size,
                callbacks=[]
            )

            last_val_loss = result.history['val_loss'][-1]

            val_losses[(learning_rate, batch_size)] = last_val_loss

            print(f"For learning rate: {learning_rate} and batch size: {batch_size}, validation loss: {last_val_loss}")

    lr_best, bs_best = min(val_losses, key=val_losses.get)

    print(f"Best learning rate: {lr_best} and best batch size: {bs_best}")
    print(f"Best validation loss: {val_losses[(lr_best, bs_best)]}")

    if save:
        print("Writing search result to 'hypertune.json'")
        hypertune_result = []
        for (learning_rate, batch_size), val_loss in val_losses.items():
            hypertune_result.append({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'val_loss': val_loss
            })
        with open(os.path.join(save, 'hypertune.json'), 'w') as f:
            json.dump(hypertune_result, f)

    print("Restoring weights")
    model.set_weights(weights)

    return lr_best, bs_best


def plot_validation_loss(json_path):

    """
    Based on 'hypertune.json', plot the validation loss as a heatmap. The plot
    will show the validation loss for each combination of learning rate and
    batch size. The X is the learning rate and the Y is the batch size. The
    color is the validation loss.
    Args:
        json_path (Path, str): Path to the JSON file containing training data.
    """

    # Read data from JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract learning rates, batch sizes, and validation losses
    learning_rates = np.array(sorted(set([d['learning_rate'] for d in data])))
    batch_sizes = np.array(sorted(set([d['batch_size'] for d in data])))
    val_losses = np.array([d['val_loss'] for d in data])

    # The vallidation loss is depicted by a rectangular colorful patch.
    # Define the patch sizes from the minimum distances between the parameter
    # values.
    patch_width = .5 * np.diff(learning_rates).min()
    patch_height = .5 * np.diff(batch_sizes).min()
    
    # #
    # Plot the graph

    def value_to_color(v, v_min, v_max, colormap=plt.cm.jet):
        normalized_value = (v - v_min) / (v_max - v_min)
        color = colormap(normalized_value)
        return color[:3]  # Return the RGB components of the color.

    fig, ax = plt.subplots()

    for d in data:
        color = value_to_color(d['val_loss'], val_losses.min(), val_losses.max())
        patch = patches.Rectangle(
            (d['learning_rate'] - patch_width / 2,
             d['batch_size'] - patch_height / 2),
            patch_width,
            patch_height,
            linewidth=1,
            edgecolor=None,
            facecolor=color
        )
        plt.text(
            d['learning_rate'],
            d['batch_size'],
            str(round(d['val_loss'], 2)),
            va='center',
            ha='center',
            color='k',
            bbox=dict(boxstyle="round",
                      ec=(1., 1., 1.),
                      fc=(1., 1., 1.))
        )
        ax.add_patch(patch)

    ax.set_xlim(learning_rates.min() - patch_width,
                learning_rates.max() + patch_width)
    ax.set_ylim(batch_sizes.min() - patch_height,
                batch_sizes.max() + patch_height)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_title('Validation Loss')

    # Show the plot
    plt.show()
