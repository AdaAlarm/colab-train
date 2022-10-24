from model_micro import make_model
import joblib
from preprocess_micro import make_data
from conf import default_conf

from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adam

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=200
    )
}

def train_evaluate(config=default_conf, save_model=False):
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1

    #print("shape:", (dx,dy))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    model = make_model(dx, dy)
    model.fit(
        X_train, y_train,
        batch_size=256, epochs=config["epochs"],
        verbose=1,
        validation_split=config['split_ratio'],
        shuffle=True
    )
    #model.summary()
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    #model.save_weights("colab-train/data/micro_model.h5")
    tf.keras.models.save_model(
        model, "colab-train/data/micro_model.h5",
        include_optimizer=False
    )

    # now prune
    model_for_pruning = sparsity.prune_low_magnitude(
        model, **pruning_params
    )

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    model_for_pruning.fit(
        X_train, y_train,
        batch_size=256, epochs=config["epochs"],
        verbose=0,
        validation_split=config['split_ratio'],
        callbacks=[
            sparsity.UpdatePruningStep(),
            sparsity.PruningSummaries(log_dir="logs/"),
        ]
    )

    score_pruning = model_for_pruning.evaluate(X_test, y_test, verbose=0)
    print('Test score (prune):', score_pruning[0])
    print('Test accuracy (prune):', score_pruning[1])

    model_for_export = sparsity.strip_pruning(model_for_pruning)
    tf.keras.models.save_model(
        model_for_export,
        "colab-train/data/micro_model_pruned.h5",
        include_optimizer=False
    )

    return score, (dx,dy)


if __name__ == '__main__':
    train_evaluate(save_model=True)

# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

