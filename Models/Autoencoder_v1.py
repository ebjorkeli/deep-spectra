from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, \
    Flatten, Layer, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

class Encoder(Layer):
    def __init__(self, filters, latent_dim, norm=False, act=None):
        super(Encoder, self).__init__()
        self.filters = filters
        self.latent_dim = latent_dim
        self.norm = norm
        self.act = act
        self.output_layer = Dense(self.latent_dim, activation='relu')

    def call(self, input):
        x = input
        # Hidden layers:
        for f in self.filters:
            x = Conv1D(filters=f, kernel_size=3, strides=2, padding='same')(x)
            if self.norm:
                x = BatchNormalization()(x)
            if self.act is not None:
                x = Activation(self.act)(x)
        x = Flatten()(x)
        #x = Dense(self.latent_dim*2, activation='relu')(x)
        x = self.output_layer(x)
        return x


class Decoder(Layer):
    def __init__(self, filters, latent_dim, norm=False, act=None):
        super(Decoder, self).__init__()
        self.filters = filters
        self.latent_dim = latent_dim
        self.norm = norm
        self.act = act
        self.output_layer = Conv1D(2, kernel_size=3, strides=1, padding='same')

    def call(self, code):
        x = Dense(1024/(2*len(self.filters))*self.filters[-1], activation='relu')(code)
        x = Reshape((-1,self.filters[-1]))(x)
        # Hidden layers:
        for f in reversed(self.filters):
            x = Conv1DTranspose(filters=f, kernel_size=3, strides=2, padding='same')(x)
            if self.norm:
                x = BatchNormalization()(x)
            if self.act is not None:
                x = Activation(self.act)(x)
        x = self.output_layer(x)
        x = Activation(self.act)(x)
        return x



# Adversary layer, currently not used
class Adversary(Layer):
    def __init__(self, l1_nodes, l2_nodes, outputs=1):
        super(Discriminator, self).__init__()
        self.l1 = Dense(l1_nodes, activation='relu', use_bias=True)
        self.l2 = Dense(l2_nodes, activation='relu', use_bias=True)
        if outputs == 1:
            act = 'sigmoid'
        else:
            act = 'softmax'
        self.output_layer = Dense(outputs, activation=act, use_bias=True)

    def call(self, code):
        x = Flatten()(code)
        x = self.l1(x)
        #x = self.l2(x)
        return self.output_layer(x)

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
    encoded = Encoder(filters=filters, latent_dim=latent_dim, norm=norm, act=activation)(input) # latent space
    out0 = Decoder(filters=filters, latent_dim=latent_dim, norm=norm, act=activation)(encoded)                         # recreated spectra

    x1 = Flatten()(encoded)
    #x2 = Flatten()(out0)

    # Is glioma (0-1):
    c0_0 = Dense(128, activation=activation, use_bias=True)(x1) # maybe print min + max
    c0_1 = BatchNormalization()(c0_0)
    out1 = Dense(1, activation='sigmoid', use_bias=True)(c0_1) # what are fed into the activations, plot or log...

    # Is mutation (0-1):xs
    #x = Dense(1024, activation='relu', use_bias=True)(x)
    #out2 = Dense(1, activation='sigmoid', use_bias=True)(x)


    # Is patient (0-5): should be adversary / discriminator
    #c1_0 = GRL()(x1)@
    c1_0 = x1
    c1_1 = Dense(128, activation=activation, use_bias=True)(c1_0)
    c1_2 = BatchNormalization()(c1_1)
    out3 = Dense(6, activation='softmax', use_bias=True)(c1_2) # want integers 0-5

    # Is machine (0-3): should be adversary / discriminator
    #c2_0 = GRL()(x1)
    c2_0 = x1
    c2_1 = Dense(128, activation=activation, use_bias=True)(c2_0)
    c2_2 = BatchNormalization()(c2_1)
    out4 = Dense(4, activation='softmax', use_bias=True)(c2_2) # want integers 0-3

    model = Model(inputs=input, outputs=[out1, out3, out4, out0, x1])
    model.summary()
    tf.keras.utils.plot_model(model, "AE_overview.png", show_shapes=True)
    return model
