from ML.preprocess import get_train_test, get_raw_train_test
from ML.model_micro import make_model
from ML.config import MLpath
import joblib

print("load data...")

try:
    print("cache?")
    d = joblib.load(MLpath + '/data/micro-cache.pkl')
    (X_train, X_test, y_train, y_test, paths_train,
     paths_test) = d[0], d[1], d[2], d[3], d[4], d[5]
    print("loaded cache")
except (IOError, FileNotFoundError):
    print("no cache ...")
    X_train, X_test, y_train, y_test, paths_train, paths_test = get_train_test(
        split_ratio=.96, sample=False)
    d = X_train, X_test, y_train, y_test, paths_train, paths_test
    joblib.dump(d, MLpath + '/data/micro-cache.pkl')

print("> done")
print(X_train.shape)

#dx = X_train.shape[1]

dx, dy, dz = 49, 40, 1


print(X_train)

X_train = X_train.reshape((X_train.shape[0], dx, dy, dz))
X_test = X_test.reshape((X_test.shape[0], dx, dy, dz))

model = make_model(dx, dy)
model.fit(
    X_train, y_train,
    batch_size=256, epochs=2, verbose=1,
    validation_data=(X_test, y_test),
    shuffle=True
)
model.summary()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights(MLpath + "/micro_model.h5")
