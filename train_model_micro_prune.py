import joblib
import tensorflow as tf
import tempfile


from model_micro import make_model
from preprocess_micro import make_data
from conf import default_conf

from tensorflow.keras.optimizers.legacy import Adam

import tensorflow_model_optimization as tfmot

def train_evaluate(config=default_conf, save_model=False):
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1
    lr = config['lr']

    print("model shape:", (dx,dy))
    print("samples:", len(X_train))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    epochs = config["epochs"]
    if config["sample"]:
        epochs = 1

    model = make_model(dx, dy, dz)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),#loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        # optimizer=Adadelta(
        #     learning_rate=1.0, rho=0.9999, epsilon=1e-08, decay=0.
        # ),
        metrics=['accuracy']
    )
    model.fit(
        X_train, y_train,
        batch_size=config["batch_size"],
        epochs=epochs,
        verbose=1,
        validation_split=1.0-config["split_ratio"]
        #shuffle=True,
    )
    #model.summary()
    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    if config["pruning"]:
        print("Begin pruning ...")

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        end_step = 844

        # Define model for pruning.
        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                       final_sparsity=0.80,
                                                                       begin_step=0,
                                                                       end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(model, **pruning_params)

        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),#loss='binary_crossentropy',
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )

        model.summary()

        print(100 * "*")

        model_for_pruning.summary()

        logdir = tempfile.mkdtemp()

        callbacks = [
          tfmot.sparsity.keras.UpdatePruningStep(),
          tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        # model_for_pruning.fit(train_images, train_labels,
        #                   batch_size=batch_size, epochs=epochs, validation_split=validation_split,
        #                   callbacks=callbacks)
        model_for_pruning.fit(
            X_train, y_train,
            batch_size=config["batch_size"],
            epochs=epochs,
            verbose=1,
            validation_split=1.0-config["split_ratio"],
            callbacks=callbacks
        )

        score = model_for_pruning.evaluate(X_test, y_test, verbose=0)

        print('Pruned Test score:', score[0])
        print('Pruned Test accuracy:', score[1])


    if save_model:
        model.save("colab-train/data/saved-model/", include_optimizer=False)
        #tf.keras.models.save_model(model, "colab-train/data/model.h5", include_optimizer=False)
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

