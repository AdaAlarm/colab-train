from train_model_micro import train_evaluate

default_conf = {
    'window_size_ms': 40,
    'window_stride_ms': 30,
    'feature_bin_count': 30,
    'epochs': 20
}


#FINALLY DONE
#[({'window_size_ms': 20, 'window_stride_ms': 10, 'feature_bin_count': 20, 'epochs': 100}, [0.34542548335138173, 0.84032923]), ({'window_size_ms': 20, 'window_stride_ms': 10, 'feature_bin_count': 30, 'epochs': 100}, [0.3121445693596891, 0.86419755]), ({'window_size_ms': 20, 'window_stride_ms': 13, 'feature_bin_count': 20, 'epochs': 100}, [0.37430717979439, 0.8131687]), ({'window_size_ms': 20, 'window_stride_ms': 13, 'feature_bin_count': 30, 'epochs': 100}, [0.47902275245376086, 0.77530867]), ({'window_size_ms': 20, 'window_stride_ms': 15, 'feature_bin_count': 20, 'epochs': 100}, [0.39019842248394654, 0.8164609]), ({'window_size_ms': 20, 'window_stride_ms': 15, 'feature_bin_count': 30, 'epochs': 100}, [0.2864746870327388, 0.8748971]), ({'window_size_ms': 25, 'window_stride_ms': 12, 'feature_bin_count': 20, 'epochs': 100}, [0.2785505375499097, 0.87983537]), ({'window_size_ms': 25, 'window_stride_ms': 12, 'feature_bin_count': 30, 'epochs': 100}, [0.2979274463506392, 0.8600823]), ({'window_size_ms': 25, 'window_stride_ms': 16, 'feature_bin_count': 20, 'epochs': 100}, [0.276835113273236, 0.8765432]), ({'window_size_ms': 25, 'window_stride_ms': 16, 'feature_bin_count': 30, 'epochs': 100}, [0.36527195084732744, 0.82386833]), ({'window_size_ms': 25, 'window_stride_ms': 18, 'feature_bin_count': 20, 'epochs': 100}, [0.3010987090721052, 0.86172837]), ({'window_size_ms': 25, 'window_stride_ms': 18, 'feature_bin_count': 30, 'epochs': 100}, [0.24746490566082943, 0.8897119]), ({'window_size_ms': 30, 'window_stride_ms': 15, 'feature_bin_count': 20, 'epochs': 100}, [0.30028360541220067, 0.873251]), ({'window_size_ms': 30, 'window_stride_ms': 15, 'feature_bin_count': 30, 'epochs': 100}, [0.32679429355962775, 0.854321]), ({'window_size_ms': 30, 'window_stride_ms': 20, 'feature_bin_count': 20, 'epochs': 100}, [0.3371687050954795, 0.8485597]), ({'window_size_ms': 30, 'window_stride_ms': 20, 'feature_bin_count': 30, 'epochs': 100}, [0.22181955257070407, 0.9111111]), ({'window_size_ms': 30, 'window_stride_ms': 22, 'feature_bin_count': 20, 'epochs': 100}, [0.2667944808678372, 0.8880658]), ({'window_size_ms': 30, 'window_stride_ms': 22, 'feature_bin_count': 30, 'epochs': 100}, [0.2475132213330563, 0.909465]), ({'window_size_ms': 40, 'window_stride_ms': 20, 'feature_bin_count': 20, 'epochs': 100}, [0.2665365499233513, 0.8839506]), ({'window_size_ms': 40, 'window_stride_ms': 20, 'feature_bin_count': 30, 'epochs': 100}, [0.24234502254199589, 0.9061728]), ({'window_size_ms': 40, 'window_stride_ms': 26, 'feature_bin_count': 20, 'epochs': 100}, [0.2774384514785107, 0.8897119]), ({'window_size_ms': 40, 'window_stride_ms': 26, 'feature_bin_count': 30, 'epochs': 100}, [0.22417074811679347, 0.9144033]), ({'window_size_ms': 40, 'window_stride_ms': 30, 'feature_bin_count': 20, 'epochs': 100}, [0.3929372532868091, 0.8320988]), ({'window_size_ms': 40, 'window_stride_ms': 30, 'feature_bin_count': 30, 'epochs': 100}, [0.20918062315801533, 0.91687244])]


if __name__ == '__main__':
    results = []

    for ws in [35, 37, 40, 42, 45, 50]:
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