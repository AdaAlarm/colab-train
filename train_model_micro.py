import joblib
import tensorflow as tf
import keras

from model_micro import make_model
from preprocess_micro import make_data
from conf import default_conf

from tensorflow.keras.utils import to_categorical



def train_evaluate(config=default_conf, save_model=False):
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']

    print("model shape:", (dx,dy))
    print("samples:", len(X_train))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    batch_size = config["batch_size"]   
    epochs = config["epochs"]

    model = make_model(dx, dy, dz)
    
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.BinaryAccuracy()
        ]
    )

    def data_generator(X, y, batch_size):
        n_samples = len(X)
        while True:  # Infinite loop for Keras
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                yield X_batch, y_batch

    train_gen = data_generator(X_train, y_train, batch_size=batch_size)
    test_gen = data_generator(X_test, y_test, batch_size=batch_size)

    model.fit(
        train_gen,
        #X_train,
        #y_train,
        #batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=1.0-config["split_ratio"]
    )

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    if save_model:
        model.save("colab-train/data/saved-model/", include_optimizer=False)
        #keras.models.save_model(model, "colab-train/data/model.h5", include_optimizer=False)
        #model.save_weights("colab-train/data/weights.tf")

    return score, (dx,dy)


if __name__ == '__main__':
    train_evaluate(save_model=True)

# (25,21):
# Test accuracy: 0.9547124


# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

