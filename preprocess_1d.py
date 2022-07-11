import glob
import numpy as np
from sklearn.model_selection import train_test_split
from adaconstants import CRYING, NOT_CRYING
from utils import models, input_data
import tensorflow as tf

MIC_RATE = 16000

try:
    import librosa
except:
    # never mind, if you don't want to make data
    pass

def float32_to_int16(y):
    """Convert float32 numpy array of audio samples to int16."""
    return (y * np.iinfo(np.int16).max).astype(np.int16)

def file_to_raw_vec(filename=None, y=None, sr=MIC_RATE):
    if y is None:
        y, sr = librosa.load(filename, sr=MIC_RATE)

    return float32_to_int16(y)


def get_data(sample=False):
    baby = glob.glob("colab-train/dataset/baby/*.wav")
    other = glob.glob("colab-train/dataset/other/*.wav")

    # allow sampling for rapid prototyping
    if sample:
        np.random.shuffle(baby)
        np.random.shuffle(other)
        baby = baby[0:50]
        other = other[0:50]

    total_files = len(baby) + len(other)

    print("files:", total_files)
    print("start preprocess...")
    c = 0

    data = []
    for f in baby:
        data.append(
            {
                "x": file_to_raw_vec(f),
                "y": CRYING,
                "path": f
            }
        )
        c += 1
        #print((c/total_files)*100, "%")
    for f in other:
        data.append(
            {
                "x": file_to_raw_vec(f),
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