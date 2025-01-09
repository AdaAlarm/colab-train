import joblib
import tensorflow as tf

from model_micro import make_model
from conf import default_conf

from tensorflow.keras.utils import to_categorical

import numpy as np
import os

def train_evaluate(X_train, X_test, y_train, y_test, config, save_model=False):
    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']

    print("model shape:", (dx,dy))
    print("samples:", len(X_train))

    # Ensure X_train and X_test are TensorFlow tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # Reshape input tensors
    X_train = tf.reshape(X_train, (tf.shape(X_train)[0], dx, dy, dz))
    X_test = tf.reshape(X_test, (tf.shape(X_test)[0], dx, dy, dz))


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
        validation_data=(X_test, y_test)
    )

    score = model.evaluate(X_test, y_test)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.export("colab-train/data/saved-model/")
    #keras.models.save_model(model, "colab-train/data/model.h5", include_optimizer=False)
    #model.save_weights("colab-train/data/weights.tf")

    return score, (dx,dy)


if __name__ == '__main__':
    conf = default_conf

    if os.path.isfile(conf['dataset_path']):
        from dataset import load_data
        (X_train, X_test, y_train, y_test, paths_train, paths_test) = load_data()
    else:
        raise Exception("dataset not found, run make_data")
    
    tf.config.run_functions_eagerly(True)
    print("Eager execution enabled:", tf.executing_eagerly())  # Should print True

    train_evaluate(X_train, X_test, y_train, y_test, conf, save_model=True)

# (25,21):
# Test accuracy: 0.9547124


# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

