from tensorflow.keras.optimizers.legacy import Adam


def warmup(model, ds, lr_list, loss, batch_size, metrics) -> None:

    """
    Warm up the model. This function trains the model one epoch
    for each learning rate value in lr_list.
    Args:
        model (keras.model): Model
        ds (tf.data.Dataset): Dataset
        lr_list: List of learning rates
        loss: Loss function, e.g. 'categorical_crossentropy'
        batch_size: Batch size
        metrics: Metrics, e.g. ['acc']
    """

    print("*** Warm up the Model")

    for learning_rate in lr_list:

        print(f"Using learning rate: {learning_rate}")

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )

        model.fit(
            x=ds,
            y=None,
            epochs=1,
            batch_size=batch_size,
            verbose=1
        )
