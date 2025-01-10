import joblib
import tensorflow as tf

from model_micro import make_model
from conf import default_conf

import tf_keras as keras

from tensorflow.keras.utils import to_categorical

import numpy as np
import os
import tempfile

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
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.BinaryAccuracy()
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

    if config["pruning"]:
        import tensorflow_model_optimization as tfmot

        print("Begin pruning ...")

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        # Define model for pruning.
        num_images = X_train.shape[0]
        end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                       final_sparsity=0.80,
                                                                       begin_step=0,
                                                                       end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(model, **pruning_params)

        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=[
                keras.metrics.BinaryAccuracy()
            ]
        )

        model.summary()

        print(100 * "*")

        model_for_pruning.summary()

        logdir = tempfile.mkdtemp()

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        model_for_pruning.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )

        score_pruned = model_for_pruning.evaluate(X_test, y_test, verbose=0)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        print('Pruned Test score:', score_pruned[0])
        print('Pruned Test accuracy:', score_pruned[1])

        model_for_pruning.export("colab-train/data/saved-pruned-model/")
        #tf.keras.models.save_model(model, "colab-train/data/model.h5", include_optimizer=False)
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

    train_evaluate(X_train, X_test, y_train, y_test, conf, save_model=True)

# (25,21):
# Test accuracy: 0.9547124


# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

