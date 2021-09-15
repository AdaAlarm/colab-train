import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Dot
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tensorflow.keras.layers import DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.regularizers import l2


def make_model(x, y, z=1):
    # softmax, 14, 4, 16 (49x20):
    # Trainable params: 4,068
    # Arena size: 34,240
    # Invoke time: ~ seconds
    # Test accuracy: .90

    # softmax, 20, 4, 30 (49x20):
    # Trainable params: 8,592
    # Arena size: >40
    # Invoke time: ~ seconds
    # Test accuracy: .91


    # softmax, 4, 6, 32 (49x40):
    # Trainable params: 58,926
    # Arena size: 29,392
    # Invoke time: ~ seconds
    # Test accuracy: 0.89

    # prelu, 4, 6, 32 (49x40):
    # Trainable params: 138,022
    # Arena size: 31,344
    # Invoke time: ~3 seconds
    # Test accuracy: 0.89
    
    nb_filters = 14  # number of convolutional filters to use
    kernel_size = (2, 2)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for pooling

    nb_layers = 4
    fully_connected = 20

    model = Sequential()
    model.add(InputLayer(input_shape=(x, y, z)))
    model.add(Conv2D(
        nb_filters,
        kernel_size=kernel_size
    ))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.add(Dropout(0.7))

    for layer in range(nb_layers):
        model.add(Conv2D(
            nb_filters,
            kernel_size=kernel_size,
            #activation='softmax',
            use_bias=False,
            padding='same'
        ))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        model.add(MaxPooling2D(pool_size=pool_size))

    #model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(AveragePooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(fully_connected, activation='softmax'))
    model.add(Dropout(0.7))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=3e-4),
        # optimizer=Adadelta(
        #     learning_rate=1.0, rho=0.9999, epsilon=1e-08, decay=0.
        # ),
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = make_model(49, 20)
    model.summary()