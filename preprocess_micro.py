from conf import default_conf
from preprocess import make_front_end
from preprocess import get_train_test
import numpy as np

def make_data(config=default_conf):

    audio_processor, model_settings = make_front_end(
        config["window_size_ms"],
        config["window_stride_ms"],
        config["feature_bin_count"]
    )

    sample = config["sample"]

    (X_train, X_test, y_train, y_test, paths_train, paths_test) = get_train_test(
        audio_processor,
        model_settings,
        split_ratio=config["split_ratio"],
        sample=sample
    )

    if config["normalize"]:
        X_train = X_train / 30.0
        X_test = X_test / 30.0

        def toint(y):
            eq = np.array_equal(y, np.array([0., 1.]))
            return int(eq)

        y_train = [toint(y) for y in y_train]
        y_test = [toint(y) for y in y_test]

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (X_train, X_test, y_train, y_test, paths_train, paths_test)


if __name__ == '__main__':
    make_data()