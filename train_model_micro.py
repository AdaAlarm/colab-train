from model_micro import make_model
import joblib
from preprocess_micro import make_data
from conf import default_conf

import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam

def train_evaluate(config=default_conf, save_model=False):
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']

    #print("shape:", (dx,dy))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    model = make_model(dx, dy, dz, lr)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        # optimizer=Adadelta(
        #     learning_rate=1.0, rho=0.9999, epsilon=1e-08, decay=0.
        # ),
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        batch_size=256,
        epochs=config["epochs"],
        verbose=1,
        validation_split=1.0-config["split_ratio"]
        #shuffle=True,
    )
    #model.summary()
    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    if save_model:
        model.save("colab-train/data/saved-model/")
        model.save_weights("colab-train/data/weights.tf")

    return score, (dx,dy)


if __name__ == '__main__':
    train_evaluate(save_model=True)

# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

