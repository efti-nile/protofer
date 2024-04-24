import json
import os

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
