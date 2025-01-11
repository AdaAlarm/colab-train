#import tensorflow as tf
import tf_keras as keras


from tf_keras import Sequential
from tf_keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Dot
from tf_keras.layers import InputLayer, Conv2D, MaxPooling2D, Lambda, BatchNormalization
from tf_keras.layers import DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D

#from tensorflow.keras.optimizers import Adadelta, Adam
from tf_keras.optimizers.legacy import Adam
from tf_keras.regularizers import l2

def make_model_orig(x, y, z=1):
    nb_filters = 24  # number of convolutional filters to use
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for pooling

    nb_layers = 4

    model = Sequential()
    model.add(InputLayer(input_shape=(x, y, z)))
    
    model.add(Conv2D(
        nb_filters,
        kernel_size=kernel_size
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for layer in range(nb_layers):
        model.add(Conv2D(
            nb_filters,
            kernel_size=kernel_size,
            use_bias=False,
            padding='same'
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    return model

    # Test accuracy: 0.9653705954551697 (2,2 kernel; 32 filters)
    # Test accuracy: 0.XX (3,3 kernel; 32 filters)
    # Test accuracy: 0.XX (3,3 kernel; 24 filters)


# gpt 11.01.2025
def make_model(x, y, z=1):
    nb_filters = 24  # Reduced filters to fit size constraints
    kernel_size = (3, 3)
    pool_size = (2, 2)
    nb_layers = 4

    model = Sequential()
    model.add(InputLayer(input_shape=(x, y, z)))

    # First Conv2D Layer
    model.add(Conv2D(
        nb_filters,
        kernel_size=kernel_size,
        padding='same',
        use_bias=False
    ))
    model.add(Activation('relu6'))
    model.add(MaxPooling2D(pool_size=pool_size))

    # Additional Conv2D Layers
    for _ in range(nb_layers - 1):
        model.add(Conv2D(
            nb_filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False
        ))
        model.add(Activation('relu6'))
        model.add(MaxPooling2D(pool_size=pool_size))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(16, activation='relu6'))
    model.add(Dense(2, activation='softmax'))

    return model



if __name__ == "__main__":
    model = make_model(21, 25)
    model.summary()
