import numpy as np
import tensorflow as tf
#from keras import backend as K
import matplotlib.pyplot as plt
from aim.tensorflow import AimCallback
#import aim

from os.path import join

path = '/Users/erinbjorkeli/phd/Data_Spectra /Real_data'
x_train, y_train = np.load(join(path, 'train/train_time_data.npy')), np.load(join(path, 'train/train_freq_data.npy'))
x_test, y_test = np.load(join(path,  'test/test_time_data.npy')), np.load(join(path,  'test/test_freq_data.npy'))
x_val, y_val = np.load(join(path,  'validate/validate_time_data.npy')), np.load(join(path,  'validate/validate_freq_data.npy'))

#x_train, y_train = x_train[:2], y_train[:2]

N, t = np.shape(x_train)

def R_squared(y, pred):
  rest = tf.reduce_sum(tf.square(tf.subtract(y, pred)))
  tot = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.divide(rest, tot))
  return r2

met = ['MeanSquaredError',
       'MeanAbsoluteError',
       'LogCoshError',
       R_squared,]
met_lab = ['mean_squared_error',
          'mean_absolute_error',
          'logcosh',
          'R_squared']

bias = True
model_auto = tf.keras.models.Sequential([
    tf.keras.layers.Dense(t*2, input_dim=t,   activation='tanh', use_bias=bias),
    tf.keras.layers.Dense(t*4, input_dim=t*2, activation='tanh', use_bias=bias),
    tf.keras.layers.Dense(t,   input_dim=t*4, activation='tanh', use_bias=bias),
    tf.keras.layers.Reshape((t,1)),
    tf.keras.layers.Conv1D(128, 5, strides=1, padding='same', activation='relu', use_bias=bias),
    tf.keras.layers.Conv1D(128, 5, strides=1, padding='same', activation='relu', use_bias=bias),
    tf.keras.layers.Conv1DTranspose(1, 7, strides=1, padding='same', use_bias=bias),
])

EPOCHS = 200
BATCH = 20

# track hyperparameters 

opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
model_auto.compile(loss='mean_squared_error', optimizer=opt, metrics=met)

hist = model_auto.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH,
                      callbacks=[AimCallback(experiment='FFT_v2')])

#hist = model_auto.fit(x_train, y_train, epochs=EPOCHS, batch_size=1,
#                      callbacks=[AimCallback(experiment='FFT_v2')])

strt, stp = 550, 800
pred = model_auto.predict(x_train)
plt.plot(y_train[0,strt:stp])
plt.plot(pred[0,strt:stp,0], alpha=0.7)
plt.show()