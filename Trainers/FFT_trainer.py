import tensorflow as tf
import numpy as np
from datetime import datetime

import aim
from aim import Run

from Utils.FFT_losses import*

class Trainer:
    def __init__(self, model, train_data, val_data, epochs, batch_size, learning_rate, aim_name='FFT_v3', save_model=True):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.N = train_data.dim[0]

        self.eps = epochs
        self.bs = batch_size
        self.lr = learning_rate

        self.name = aim_name
        self.save = save_model

### Loss functions
    def training_loss(self, prediction, target):
        loss_gradient = 0
        loss = mean_squared(target, prediction)
        return loss

    def validation_loss(self, prediction, target):
        loss = mean_squared(target, prediction)
        return loss

### Step functions
    def train_stp(self, epoch, optimizer):
        data = next(self.train_data.next_batch(self.bs))
        with tf.GradientTape() as tape:
            prediction = self.model(data[0], training=False)
            loss = self.training_loss(prediction[...,0], data[1])

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def validation_stp(self, epoch):
        data = next(self.val_data.next_batch(self.bs))
        prediction = self.model(data[0], training=False)
        loss = self.validation_loss(prediction[...,0], data[1])
        return loss

### Training function
    def train(self,):
        my_run = Run(experiment=self.name, )
        run['hparams'] = {'epochs': self.eps,
                          'batch_size': self.bs,
                          'learning_rate': self.lr, }

        loss = np.zeros((2, self.eps))
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for epoch in range(self.eps):
            if epoch % 10 == 0:
                aim.track(loss.item(), name='loss', epoch=epoch, subset='train')
            aim.track(loss.item(), name='loss', epoch=epoch, subset='val')

            for step in range(self.N//self.bs):
                train_loss = self.train_stp(epoch,opt)
                val_loss = self.validation_stp(epoch)

            loss[0,epoch], loss[1,epoch] = train_loss, val_loss
        if self.save:
            self.model.save('{}'.format(self.name + str(datetime.now())))

        return loss
