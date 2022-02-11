from model_1d import make_model
import joblib

print("load data...")

try:
    print("cache?")
    d = joblib.load('colab-train/data/1d-cache.pkl')
    (X_train, X_test, y_train, y_test, paths_train,
     paths_test) = d[0], d[1], d[2], d[3], d[4], d[5]
    print("loaded cache")
except (IOError, FileNotFoundError):
    print("no cache ...")

dx = 16000

X_train = X_train.reshape((X_train.shape[0], dx))
X_test = X_test.reshape((X_test.shape[0], dx))

model = make_model(dx)
model.fit(
    X_train, y_train,
    batch_size=256, epochs=500, verbose=1,
    validation_data=(X_test, y_test),
    shuffle=True
)
model.summary()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights("colab-train/data/1d_model.h5")

# 09.02.22
# loss: 0.1043 - accuracy: 0.9684 - val_loss: 0.1733 - val_accuracy: 0.9439
# 11.02.22
# loss: 0.0791 - accuracy: 0.9815 - val_loss: 0.1800 - val_accuracy: 0.9505