import tensorflow as tf
import numpy as np

#from firmware.config import mic_rate
#from ML.model import make_model
from ML.model_micro import make_model


model = make_model(49, 40)
model.load_weights("ML/micro_model.h5")
model.summary()

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# this is doable with tensorflow==2.5.0
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

def representative_dataset():
    import glob
    import random
    from ML.preprocess import file_to_vec

    path1 = glob.glob("/Users/askemottelson/Dropbox/babyalarm/ada-alarm/ML/data/dataset.1s.16k/baby/*.wav")
    path2 = glob.glob("/Users/askemottelson/Dropbox/babyalarm/ada-alarm/ML/data/dataset.1s.16k/other/*.wav")

    paths = random.sample(path1+path2, 1000)

    for path in paths[0:100]:
        vec = file_to_vec(path).reshape(1, 49, 40, 1)
        yield [vec]

converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8


tflite_model = converter.convert()
open("ML/lite/micro-model.tflite","wb").write(tflite_model)

