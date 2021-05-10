from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from baselines import create_cnn, create_lstm
from utils import denormalize, process_data


def predict(model, x, true_y, scaler):
    true_y = np.squeeze(true_y, 1)
    preds = model.predict(x)

    preds = np.array(preds)
    mse = np.mean((preds - true_y) ** 2)
    # mse = mean_squared_error(true_y, preds)

    # Denormalize before plot and mse
    preds = scaler.inverse_transform(preds)
    true_y = scaler.inverse_transform(true_y)

    # Plot preds and true
    for i in range(preds.shape[1]):
        plt.plot([j for j in range(len(preds))], preds[:, i].ravel(), label="Preds")
        plt.plot([j for j in range(len(true_y))], true_y[:, i].ravel(), label="True")
        plt.title(f"Feature: {i}")
        plt.legend()
        plt.show()

    return mse


if __name__ == "__main__":
    # ---  Settings ---#
    # General
    window_size = 50
    horizon = 1
    activation = "relu"
    lr = 1e-5
    dropout = 0.0
    epochs = 30
    batch_size = 32

    # LSTM
    lstm_units = 128

    # CNN
    filter_size = 7
    filters = 64

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "cnn"])
    args = parser.parse_args()

    data = process_data(window_size, horizon)
    feature_names = data["feature_names"]

    print(feature_names)

    train_x = data["train_x"]
    train_y = data["train_y"]

    val_x = data["val_x"]
    val_y = data["val_y"]

    test_x = data["test_x"]
    test_y = data["test_y"]

    scaler = data["scaler"]

    feature_dim = len(feature_names)

    if args.model == "lstm":
        model = create_lstm(
            train_x.shape[1:],
            lstm_units,
            activation=activation,
            dropout=dropout,
            out_dim=feature_dim,
        )
    else:
        model = create_cnn(
            train_x.shape[1:],
            activation=activation,
            kernel_size=filter_size,
            dropout=dropout,
            filters=filters,
            out_dim=feature_dim,
        )

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mse", optimizer=opt)

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[],
    )

    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    # Predict
    mse_train = predict(model, train_x, train_y, scaler)
    mse_val = predict(model, val_x, val_y, scaler)
    mse_test = predict(model, test_x, test_y, scaler)
