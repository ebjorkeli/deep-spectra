from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, Flatten
from tensorflow.keras.models import Model

def model_1(input_shape):
    input = Input((input_shape))
    x  = Dense(t*2, input_dim=t,   activation='tanh', use_bias=True)(input)
    x  = Dense(t*2, input_dim=t**2,   activation='tanh', use_bias=True)(x)
    x = Dense(t*2, input_dim=t**2,   activation='tanh', use_bias=True)(x)
    x = Reshape((t,1))(x)
    x = Conv1D(128, 5, strides=1, padding='same', activation = 'relu', use_bias=True)(x)
    x = Conv1D(128, 5, strides=1, padding='same', activation='relu', use_bias=True)(x)
    output = Conv1DTranspose(1, 7, strides=1, padding='same', use_bias=True)(x)

    model = Model(inputs=[input], outputs=[output])
    model.summary()
    return model