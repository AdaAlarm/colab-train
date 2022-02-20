from train_model_micro import train_evaluate

default_conf = {
    'window_size_ms': 40,
    'window_stride_ms': 30,
    'feature_bin_count': 30,
    'epochs': 20
}




if __name__ == '__main__':
    results = []

    for ws in [35, 40, 35]:
        window_size = ws

        for x in [2/3, 3/4, 4/5]:
            window_stride_ms = int(window_size*x)

            for fb in [25, 30, 35]:
                feature_bin_count = fb

                conf = {
                    'window_size_ms': window_size,
                    'window_stride_ms': window_stride_ms,
                    'feature_bin_count': feature_bin_count,
                    'epochs': 200
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