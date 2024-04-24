from tensorflow.keras.optimizers.legacy import Adam


def train(model, train_ds, val_ds, learning_rate, loss, epochs, batch_size, callbacks):

    """
    Train the model.
    Args:
        model (keras.model): Model
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        learning_rate: Learning rate
        loss (str): Loss function (e.g. 'categorical_crossentropy')
        epochs: Number of epochs
        batch_size: Batch size
        callbacks (list of keras.callbacks.Callback): Callbacks
    Returns:
        result (keras.callbacks.History): History
    """

    print("*** Train the Model")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['acc']
    )

    result = model.fit(
        x=train_ds,
        y=None,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_ds,
        verbose=1,
        callbacks=callbacks
    )

    return result
