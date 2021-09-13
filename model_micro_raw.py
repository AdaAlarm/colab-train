import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Dot
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tensorflow.keras.layers import DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.regularizers import l2


def make_model(x,y=1,z=1):
    nb_filters = 14  # number of convolutional filters to use
    kernel_size = (3, 1)  # convolution kernel size
    pool_size = (2, 1)  # size of pooling area for pooling

    nb_layers = 4
    fully_connected = 20

    model = Sequential()
    model.add(InputLayer(input_shape=(x,y,z)))
    model.add(Conv2D(
        nb_filters,
        kernel_size=kernel_size
    ))
    model.add(BatchNormalization())
    model.add(Activation('PReLU'))

    for layer in range(nb_layers):
        model.add(Conv2D(
            nb_filters,
            kernel_size=kernel_size,
            #activation='PReLU',
            use_bias=False,
            padding='same'
        ))
        model.add(BatchNormalization())
        model.add(Activation('PReLU'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.5))

    #model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(AveragePooling2D(pool_size=pool_size))

    model.add(Flatten())

    model.add(Dense(fully_connected, activation='PReLU'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='PReLU'))
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
    model = make_model(16000)
    model.summary()
