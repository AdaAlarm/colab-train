import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adadelta, Adam

def make_model(x, y, z=1):
    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(x/2), int(y/2))
    
    # Model layers
    # Input pure conv2d
    inputs = Input(shape=(x,y,z))
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.3)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = AveragePooling2D(pool_size=final_pool_size)(x)
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    outputs = Dense(2, activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

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
