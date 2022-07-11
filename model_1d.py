import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import InputLayer, Conv1D, MaxPool1D, BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.regularizers import l2

import numpy as np


def extract_features(inputs):
    sample_rate = 16000.0

    # convert to float 32, which stft will want
    inputs = tf.cast(inputs, tf.float32)

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(inputs, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    #lower_edge_hertz, upper_edge_hertz, num_mel_bins = 125.0, 7500.0, 60
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    
    # take the first 64
    return log_mel_spectrograms[..., :64]


def make_model(raw_size):

    nb_filters = 64  # number of convolutional filters to use
    kernel_size = 3  # convolution kernel size
    pool_size = 2  # size of pooling area for max pooling

    nb_layers = 4

    regularizer = l2(1e-3)

    model = Sequential()

    model.add(InputLayer(input_shape=raw_size, dtype=tf.int16))
  
    model.add(Lambda(extract_features))

    model.add(Dense(nb_filters))
    model.add(Conv1D(
        nb_filters,
        kernel_size=kernel_size,
        kernel_regularizer=regularizer,
        activation='relu',
        use_bias=False
    ))
    model.add(BatchNormalization(axis=1, fused=False))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for layer in range(nb_layers):
        model.add(Conv1D(
            nb_filters,
            kernel_size=kernel_size,
            kernel_regularizer=regularizer,
            use_bias=False
        ))
        model.add(BatchNormalization(axis=1, fused=False))
        model.add(Activation('relu'))
        model.add(MaxPool1D(pool_size=pool_size))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nb_filters * nb_layers, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='binary_crossentropy',
        # optimizer=Adadelta(
        #     learning_rate=0.99, rho=0.9999, epsilon=1e-08, decay=0.
        # ),
        optimizer=Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    return model



if __name__ == "__main__":
    model = make_model(16000)
    model.summary()