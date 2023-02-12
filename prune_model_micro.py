from model_micro import make_model
import joblib
from preprocess_micro import make_data
from conf import default_conf

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow.keras.optimizers import Adadelta, Adam


def train_evaluate(config=default_conf, save_model=False):
    # this should be faster than training, because data already is preprocessed
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1

    #print("shape:", (dx,dy))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    model = tf.keras.models.load_model('colab-train/data/saved-model/')

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)
    #model_for_pruning.summary()

    log_dir = "logs/"
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Log sparsity and other metrics in Tensorboard.
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]

    model_for_pruning.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    model_for_pruning.fit(
        X_train, y_train,
        callbacks=callbacks,
        epochs=config['epochs'],
    )
    #model_for_pruning.summary()
    score = model_for_pruning.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    if save_model:
        model_for_pruning.save("colab-train/data/saved-model-pruned/", include_optimizer=True)

    return score, (dx,dy)


if __name__ == '__main__':
    train_evaluate(save_model=True)

# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

