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
    
    batch_size = config["batch_size"]   
    epochs = config["prune_epochs"]    

    model = keras.saving.load_model("colab-train/data/saved-model/")
    

    if config["pruning"]:
        import tensorflow_model_optimization as tfmot

        print("Begin pruning ...")

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        # Define model for pruning.
        num_images = X_train.shape[0]
        end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs * 0.9

        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.15,
                                                                       final_sparsity=0.60,
                                                                       begin_step=0,
                                                                       end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(model, **pruning_params)

        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=lr/10),
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

        print('Pruned Test score:', score_pruned[0])
        print('Pruned Test accuracy:', score_pruned[1])

        # STRIP
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        model_for_export.save("colab-train/data/saved-pruned-model/")


    return score_pruned, (dx,dy)


if __name__ == '__main__':
    conf = default_conf

    if os.path.isfile(conf['dataset_path']):
        from dataset import load_data
        (X_train, X_test, y_train, y_test, paths_train, paths_test) = load_data()
    else:
        raise Exception("dataset not found, run make_data")
    
    tf.config.run_functions_eagerly(True)

    train_evaluate(X_train, X_test, y_train, y_test, conf, save_model=True)

