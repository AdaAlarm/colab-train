import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Dot
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tensorflow.keras.layers import DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.regularizers import l2

from tensorflow_model_optimization.sparsity import keras as sparsity

pruning_schedule = sparsity.PolynomialDecay(
    initial_sparsity=0.0, final_sparsity=0.5,
    begin_step=1000, end_step=15000
)

def make_model(x, y, z=1):
    nb_filters = 24  # number of convolutional filters to use
    kernel_size = (2, 2)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for pooling

    nb_layers = 4
    #fully_connected = 20

    model = Sequential()
    model.add(InputLayer(input_shape=(x, y, z)))
    
    #for layer in range(nb_layers-1):
    model.add(Conv2D(
        nb_filters,
        kernel_size=kernel_size
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for layer in range(nb_layers):
        model.add(
            sparsity.prune_low_magnitude(
                Conv2D(
                    nb_filters,
                    kernel_size=kernel_size,
                    #activation='softmax',
                    #kernel_regularizer=lr,
                    use_bias=False,
                    padding='same'
                ),
                pruning_schedule=pruning_schedule
            )
        )

        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))#, padding="same"))

    #model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(AveragePooling2D(pool_size=pool_size))

    model.add(Flatten())

    #model.add(Dense(fully_connected, activation='softmax'))
    model.add(Dropout(0.5))
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
    model = make_model(21, 25)
    model.summary()
