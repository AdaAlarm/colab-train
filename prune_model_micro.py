from model_micro import make_model
import joblib
from preprocess_micro import make_data
from conf import default_conf

import tensorflow as tf
import tensorflow_model_optimization as tfmot


def train_evaluate(config=default_conf, save_model=False):
    # this should be faster than training, because data already is preprocessed
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1

    #print("shape:", (dx,dy))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    model = make_model(dx, dy)
    model.load_weights("colab-train/data/micro_model.h5")

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)
    model_for_pruning.summary()

    log_dir = "logs/"
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Log sparsity and other metrics in Tensorboard.
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]

    model_for_pruning.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    model_for_pruning.fit(
        X_train, y_train,
        callbacks=callbacks,
        epochs=2,
    )
    #model.summary()
    score = model_for_pruning.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    if save_model:
        model_for_pruning.save("colab-train/data/micro_model_pruned.h5", include_optimizer=True)

    return score, (dx,dy)


if __name__ == '__main__':
    train_evaluate(save_model=True)

# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

