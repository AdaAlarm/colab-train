import joblib
import tensorflow as tf

from model_micro import make_model
from preprocess_micro import make_data
from conf import default_conf

from tensorflow.keras.utils import to_categorical

import numpy as np


@tf.function
def train_evaluate(X_train, X_test, y_train, y_test, config, save_model=False):
    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']

    print("model shape:", (dx,dy))
    print("samples:", len(X_train))

    # Reshape input tensors
    X_train = tf.reshape(X_train, (tf.shape(X_train)[0], dx, dy, dz))
    X_test = tf.reshape(X_test, (tf.shape(X_test)[0], dx, dy, dz))

    # Ensure labels are tensors
    y_train = tf.cast(y_train, tf.int32)
    y_test = tf.cast(y_test, tf.int32)

    # Ensure labels are 1D (batch_size,) before one-hot encoding
    if len(y_train.shape) > 1:
        y_train = tf.squeeze(y_train, axis=-1)
    if len(y_test.shape) > 1:
        y_test = tf.squeeze(y_test, axis=-1)

    # One-hot encode the labels
    y_train = tf.one_hot(y_train, depth=2)
    y_test = tf.one_hot(y_test, depth=2)

    print(y_train.shape)
    print(y_test.shape)
    print(70 * "*")
    
    batch_size = config["batch_size"]   
    epochs = config["epochs"]

    model = make_model(dx, dy, dz)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            tf.keras.metrics.BinaryAccuracy()
        ]
    )

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save("colab-train/data/saved-model/", include_optimizer=False)
    #keras.models.save_model(model, "colab-train/data/model.h5", include_optimizer=False)
    #model.save_weights("colab-train/data/weights.tf")

    return score, (dx,dy)


if __name__ == '__main__':
    conf = default_conf

    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(conf)

    train_evaluate(X_train, X_test, y_train, y_test, conf, save_model=True)

# (25,21):
# Test accuracy: 0.9547124


# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

