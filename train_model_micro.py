from model_micro import make_model
import joblib

print("load data...")

try:
    print("cache?")
    d = joblib.load('colab-train/data/micro-cache.pkl')
    (X_train, X_test, y_train, y_test, paths_train,
     paths_test) = d[0], d[1], d[2], d[3], d[4], d[5]
    print("loaded cache")
except (IOError, FileNotFoundError):
    print("no cache ...")

print("> done")
print(X_train.shape)

#dx = X_train.shape[1]

dx, dy, dz = 65, 20, 1


X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

model = make_model(dx, dy)
model.fit(
    X_train, y_train,
    batch_size=256, epochs=1000, verbose=1,
    validation_data=(X_test, y_test),
    shuffle=True
)
model.summary()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights("colab-train/data/micro_model.h5")


# (49,20):
# Test accuracy: 0.9279835224151611

# (49,40):
# Test accuracy: 0.9331275820732117

