import joblib
import tensorflow as tf

from model_micro import make_model
from conf import default_conf

import tf_keras as keras

import numpy as np
import os
import tempfile

def train_evaluate(X_train, X_test, y_train, y_test, config, save_model=False):
    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']
    
    batch_size = config["batch_size"]

    if config["pruning"]:
        import tensorflow_model_optimization as tfmot

        model_for_export = keras.saving.load_model("colab-train/data/saved-pruned-model/")

        # PQAT
        quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
            model_for_export
        )
        pqat_model = tfmot.quantization.keras.quantize_apply(
            quant_aware_annotate_model,
            tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme()
        )

        pqat_model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=lr/10),
            metrics=[
                keras.metrics.BinaryAccuracy()
            ]
        )

        print('Train pqat model:')
        pqat_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=(X_test, y_test)
        )

        score_pqat = pqat_model.evaluate(X_test, y_test, verbose=0)

        print('PQAT Test score:', score_pqat[0])
        print('PQAT Test accuracy:', score_pqat[1])

        pqat_model.save("colab-train/data/pqat_model/")


    return score_pqat, (dx,dy)


if __name__ == '__main__':
    conf = default_conf

    if os.path.isfile(conf['dataset_path']):
        from dataset import load_data
        (X_train, X_test, y_train, y_test, paths_train, paths_test) = load_data()
    else:
        raise Exception("dataset not found, run make_data")
    
    tf.config.run_functions_eagerly(True)

    train_evaluate(X_train, X_test, y_train, y_test, conf, save_model=True)

