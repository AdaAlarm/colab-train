from model_1d import make_model
import joblib
from preprocess_1d import get_train_test
from conf import default_conf

def train_evaluate(config=default_conf, save_model=False):
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = get_train_test(split_ratio=.95)

    dx = 16000

    #epochs = config["epochs"]
    epochs = 250

    model = make_model(dx)
    model.fit(
        X_train, y_train,
        batch_size=256, epochs=epochs, verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True
    )
    #model.summary()
    score = model.evaluate(X_test, y_test, verbose=0)
    
    if save_model:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights("colab-train/data/1d_model.h5")

    return score, dx


if __name__ == '__main__':
    train_evaluate(save_model=True)


