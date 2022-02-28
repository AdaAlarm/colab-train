from model_micro import make_model
import joblib
from preprocess_micro import make_data
from conf import default_conf

def train_evaluate(config=default_conf, save_model=False):
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(config)

    dx, dy, dz = X_train.shape[1], X_train.shape[2], 1

    #print("shape:", (dx,dy))

    X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
    X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

    model = make_model(dx, dy)
    model.fit(
        X_train, y_train,
        batch_size=256, epochs=config["epochs"], verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True
    )
    #model.summary()
    score = model.evaluate(X_test, y_test, verbose=0)
    
    if save_model:
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights("colab-train/data/micro_model.h5")

    return score, (dx,dy)


if __name__ == '__main__':
    train_evaluate(save_model=True)

# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

# (22,30)

