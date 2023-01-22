from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, \
    Flatten, Layer, BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] == self.units
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


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
    #encoded, out0 = AutoEncoder(filters=filters, latent_dim=latent_dim, norm=norm, act=activation)(input)

    ###
    ###
    xs = {}
    xs[0] = input
    # Encode:
    for i in range(len(filters)):  # enumerate outputs x1, x2...
        x = Conv1D(filters=filters[i], kernel_size=3, strides=2, padding='same')(xs[i])
        if norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
            xs[i + 1] = x
        print('down', x.shape)
    x = Flatten()(x)
    encoded = Dense(latent_dim, activation='relu')(x) # about this one and flatten, very large...
    print('code', encoded.shape)

    # Decode:
    x = Dense(1024 / (2 * len(filters)) * filters[-1], activation='relu')(encoded)
    print('decode', x.shape)
    x = Reshape((-1, filters[-1]))(x)
    for i in reversed(range(len(filters))):
        x = concatenate([x, xs[i + 1]])  # seems ok
        x = Conv1DTranspose(filters=filters[i], kernel_size=3, strides=2, padding='same')(x)
        if norm:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        print('up', x.shape)
    x = Conv1D(2, kernel_size=3, strides=1, padding='same')(x)
    out0 = Activation(activation)(x)

    ###

    x1 = Flatten()(encoded)
    #x2 = Flatten()(out0)

    # Is glioma (0-1):
    c0_0 = Dense(128, activation=activation, use_bias=True)(x1) # maybe print min + max
    c0_1 = BatchNormalization()(c0_0)
    out1 = Dense(1, activation='sigmoid', use_bias=True)(c0_1) # what are fed into the activations, plot or log...

    #Is mutation (0-1):xs
    c_3_0 = Dense(1024, activation='relu', use_bias=True)(x)
    out2 = Dense(1, activation='sigmoid', use_bias=True)(x)


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
