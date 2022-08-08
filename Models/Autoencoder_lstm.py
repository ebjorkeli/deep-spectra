from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, \
    Flatten, Layer, BatchNormalization, Activation, LSTM, TimeDistributed, RepeatVector, Permute
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# Gradient reversal layer
@tf.custom_gradient
def custom_op(x): # is it working correctly?
    y = tf.identity(x)
    def reverse_grad(dy):
        return tf.negative(dy)
    return y, reverse_grad

class GRL(Layer):
    def __init__(self):
        super(GRL, self).__init__()

    def call(self, inputs):
        return custom_op(inputs)

# Full model
def autoencodeclass(input_shape, config):
    filters = config['filters']
    latent_dim = config['latent_dim']
    norm = config['batch_normalization']
    activation = 'elu'

    input = Input((input_shape))

    # Autoencoder
    x = Permute((2, 1))(input)
    # Hidden layers:
    for f in filters[:-1]:
        x = LSTM(f, return_sequences=True)(x)
    encoded = LSTM(filters[-1], return_sequences=False)(x)  # encoded

    x = RepeatVector(2)(encoded)
    # Hidden layers:
    for f in reversed(filters):
        x = LSTM(f, return_sequences=True)(x)
    x = LSTM(1024, return_sequences=True)(x)
    out0 = Permute((2, 1))(x)

    #out0 = Decoder(filters=filters, latent_dim=latent_dim, norm=norm, act=activation)(encoded)                         # recreated spectra

    x1 = Flatten()(encoded)
    #x2 = Flatten()(out0)

    # Is glioma (0-1):
    c0_0 = BatchNormalization()(x1, training=True)
    c0_0 = Dense(128, activation=activation, use_bias=True)(c0_0) # maybe print min + max
    c0_1 = BatchNormalization()(c0_0, training=True)
    out1 = Dense(1, activation='sigmoid', use_bias=True)(c0_1) # what are fed into the activations, plot or log...

    # Is mutation (0-1):xs
    #x = Dense(1024, activation='relu', use_bias=True)(x)
    #out2 = Dense(1, activation='sigmoid', use_bias=True)(x)


    # Is patient (0-5): should be adversary / discriminator
    c1_0 = GRL()(x1)
    c1_0 = BatchNormalization()(c1_0, training=True)
    c1_1 = Dense(128, activation=activation, use_bias=True)(c1_0)
    c1_2 = BatchNormalization()(c1_1, training=True)
    out3 = Dense(6, activation='softmax', use_bias=True)(c1_2) # want integers 0-5

    # Is machine (0-3): should be adversary / discriminator

    c2_0 = GRL()(x1)
    c2_0 = BatchNormalization()(c2_0, training=True)
    c2_1 = Dense(128, activation=activation, use_bias=True)(c2_0)
    c2_2 = BatchNormalization()(c2_1, training=True)
    out4 = Dense(4, activation='softmax', use_bias=True)(c2_2) # want integers 0-3

    #model = Model(inputs=input, outputs=out0)
    model = Model(inputs=input, outputs=[out1, out3, out4, out0, x1, c0_0, c0_1])
    #model.summary()
    tf.keras.utils.plot_model(model, "AE_overview.png", show_shapes=True)
    return model
