from preprocess_micro import make_data
from conf import default_conf

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test, paths_train, paths_test) = make_data(default_conf)