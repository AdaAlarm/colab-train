from preprocess import get_train_test
from model1d import make_model

import joblib
from train_model_micro import default_conf
from preprocess import make_front_end


def make_data(conf=default_conf):

    audio_processor = make_front_end(
        conf["window_size_ms"],
        conf["window_stride_ms"],
        conf["feature_bin_count"]
    )

    return get_train_test(split_ratio.95)


if __name__ == '__main__':
    make_data()