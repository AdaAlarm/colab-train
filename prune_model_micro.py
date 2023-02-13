from model_micro import make_model
import joblib
from preprocess_micro import make_data
from conf import default_conf

import tensorflow as tf
import tensorflow_model_optimization as tfmot

#from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow import keras


import tempfile

def train_evaluate(config=default_conf, save_model=False):
    # this should be faster than training, because data already is preprocessed
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']

    #print("shape:", (dx,dy))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    model = tf.keras.models.load_model('colab-train/data/saved-model/')
    #model = make_model(dx, dy, dz, lr)
    #model.load_weights("colab-train/data/weights.tf")
    #model = tf.keras.models.load_model("colab-train/data/model.h5")

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)
    #model_for_pruning.summary()

    model_for_pruning.compile(
        #loss='binary_crossentropy',
        loss=tf.keras.losses.binary_crossentropy,
        optimizer='adam',
        #optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    log_dir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Log sparsity and other metrics in Tensorboard.
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]
    model_for_pruning.fit(
        X_train, y_train,
        batch_size=256,
        epochs=2,
        verbose=1,
        validation_split=1.0-config["split_ratio"],
        callbacks=callbacks
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
