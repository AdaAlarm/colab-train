from train_model_micro import train_evaluate

default_conf = {
    'window_size_ms': 30,
    'window_stride_ms': 20,
    'feature_bin_count': 20,
    'epochs': 20
}




if __name__ == '__main__':
    results = []

    for ws in [20, 25, 30, 40]:
        window_size = ws

        for x in [1/2, 2/3, 3/4]:
            window_stride_ms = int(window_size*x)

            for fb in [15, 20, 30]:
                feature_bin_count = fb

                conf = {
                    'window_size_ms': window_size,
                    'window_stride_ms': window_stride_ms,
                    'feature_bin_count': feature_bin_count,
                    'epochs': 100
                }

                try:
                    score = train_evaluate(conf)
                except:
                    print("config doesn't work")
                    print(conf)
                    continue

                print(70 * "*")
                print(conf)
                print('Test score:', score[0])
                print('Test accuracy:', score[1])
                print(70 * "*")

                results.append((conf, score))

    print("FINALLY DONE")
    print(results)