import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from conf import default_conf

def load_data():
    pickle_file = default_conf['dataset_path']

    # load existing preprocessed data
    with open(pickle_file, 'rb') as handle:
        print("loaded data cache.")
        data = pickle.load(handle)

    X = np.array([d["x"] for d in data])
    y = np.array([d["y"] for d in data])
    files = [d["path"] for d in data]

    random_state = 42

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, files,
        test_size=(1 - default_conf["split_ratio"]),
        random_state=random_state,
        shuffle=True
    )

    return X_train, X_test, y_train, y_test, paths_train, paths_test