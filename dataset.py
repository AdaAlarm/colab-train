

def load_data():
    pickle_file = 'colab-train/data/data.pickle'
    try:
        # load existing preprocessed data
        with open(pickle_file, 'rb') as handle:
            print("loaded data cache.")
            data = pickle.load(handle)

    X = np.array([d["x"] for d in data])
    y = np.array([d["y"] for d in data])

    return X, y, [d["path"] for d in data]