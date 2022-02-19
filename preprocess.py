import glob
import numpy as np
from sklearn.model_selection import train_test_split
from adaconstants import CRYING, NOT_CRYING
from utils import models, input_data
import tensorflow as tf

sess = tf.compat.v1.InteractiveSession()


def make_front_end(size=30, stride=20, bins=20):

    model_settings = models.prepare_model_settings(
        label_count=2,
        sample_rate=16000,
        clip_duration_ms=1000,
        window_size_ms=size, #30
        window_stride_ms=stride, #20
        feature_bin_count=bins, #20
        preprocess='micro'
    )

    audio_processor = input_data.AudioProcessor(
        None, None, 0, 0, '', 0, 0, model_settings, None
    )

    return audio_processor, model_settings


def file_to_vec(audio_processor, model_settings, filename=None):

    results = audio_processor.get_features_for_wav(
        filename, model_settings, sess
    )

    return results[0]


def get_data(audio_processor, model_settings, sample=False):
    baby = glob.glob("colab-train/dataset/baby/*.wav")
    other = glob.glob("colab-train/dataset/other/*.wav")

    # allow sampling for rapid prototyping
    if sample:
        np.random.shuffle(baby)
        np.random.shuffle(other)
        baby = baby[0:50]
        other = other[0:50]

    total_files = float(len(baby) + len(other))

    print("files:", total_files)
    print("start preprocess...")
    c = 0

    data = []
    for f in baby:
        data.append(
            {
                "x": file_to_vec(audio_processor, model_settings, f),
                "y": CRYING,
                "path": f
            }
        )
        c += 1
        #print((c/total_files)*100, "%")
    for f in other:
        data.append(
            {
                "x": file_to_vec(audio_processor, model_settings, f),
                "y": NOT_CRYING,
                "path": f
            }
        )
        c += 1
        #print((c/total_files)*100, "%")

    print("preprocess done")

    np.random.shuffle(data)

    X = np.array([d["x"] for d in data])
    y = np.array([d["y"] for d in data])

    return X, y, [d["path"] for d in data]


def return_train_test(split_ratio, X, y, files, micro, sample):
    random_state = 42

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, files,
        test_size=(1 - split_ratio),
        random_state=random_state,
        shuffle=True
    )

    return X_train, X_test, y_train, y_test, paths_train, paths_test


def get_train_test(audio_processor=None, model_settings=None, split_ratio=0.8, sample=False):
    if audio_processor is None:
        # default
        audio_processor, model_settings = make_front_end()

    X, y, files = get_data(
        audio_processor=audio_processor,
        model_settings=model_settings,
        sample=sample
    )

    assert X.shape[0] == len(y)

    return return_train_test(split_ratio, X, y, files, True, sample)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()

    print(X_train)
    print(y_train)
    print(X_train.shape)
