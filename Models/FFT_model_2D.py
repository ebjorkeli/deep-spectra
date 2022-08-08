from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, Flatten
from tensorflow.keras.models import Model

def model_1(input_shape, output_shape):
    input = Input((input_shape))
    x = Flatten()(input)
    x  = Dense(1024, input_dim=2048,   activation='tanh', use_bias=True)(x)
    x  = Dense(1024, input_dim=1024,   activation='tanh', use_bias=True)(x)
    x = Dense(1024, input_dim=1024,   activation='tanh', use_bias=True)(x)
    x = Reshape((output_shape))(x)
    x = Conv1D(128, 5, strides=1, padding='same', activation = 'relu', use_bias=True)(x)
    x = Conv1D(128, 5, strides=1, padding='same', activation='relu', use_bias=True)(x)
    output = Conv1DTranspose(1, 7, strides=1, padding='same', use_bias=True)(x)

    model = Model(inputs=[input], outputs=[output]) # input shape (1024,2) output shape (1024,1)
    model.summary()
    return model