# Something that judges if a recreated spectra is more probable to bre real than comparable real spectra

# Pretrain a model to judge if input is a real spectra...

from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, Conv1DTranspose, \
    Flatten, Layer, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow as tf

class evaluator_model(Model):
    def __init__(self):
        super(evaluator_model, self).__init__()
        self.l1 = Dense(512)
        self.l2 = Dense(128)
        self.l3 = Dense(1)
        self.act = Activation('sigmoid')

    def call(self, input):
        x = Input(input)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.act(x)


class realness_eval():
    def __init__(self):
        try:
            self.model = tf.load('pretrained_spectra_real_eval')
        except:
            print('could not find pretrained model...')
            # Set up a new model
            self.model = construct_model()
            # Train a new model
            self.model.
            # Save the model
            model.save('pretrained_spectra_real_eval')


    def set_up_model(self):
        model = evaluator_model()
        return model

    def train_model(self):
        return model

    def is_real(self, input):
        ## evaluate realness
        prob = self.model(input, training=False)