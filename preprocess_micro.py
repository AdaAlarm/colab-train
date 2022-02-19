from conf import default_conf
from preprocess import make_front_end
from preprocess import get_train_test

def make_data(conf=default_conf):

    audio_processor, model_settings = make_front_end(
        conf["window_size_ms"],
        conf["window_stride_ms"],
        conf["feature_bin_count"]
    )

    return get_train_test(audio_processor, model_settings, split_ratio=.95)


if __name__ == '__main__':
    make_data()