import tensorflow as tf
import numpy as np
from os.path import join

from Utils.FFT_losses import*
from Utils.evaluation import evaluate, accuracy
from Utils.tsne import tSNE
from Utils.confusion_matrix import CM

class Trainer:
    def __init__(self, model, train_data, val_data, config):
        self.config      = config
        self.model       = model
        self.model_type  = config['model']
        self.train_data  = train_data
        self.val_data    = val_data
        self.num_spec    = train_data.dim[0]

        self.eps = config['max_epochs']
        self.bs  = config['batch_size']
        self.lr  = config['learning_rate']
        self.ur  = config['update_ratio']
        self.lambda_gp = config['lambda_gp']

        self.name = config['run_name']
        self.time = config['datetime']

        self.w = config['loss_weights'] #weights for losses (glioma, PID, MID, recreated spectra)

        self.path = config['data_path']

### Loss functions
    def training_loss(self, prediction, target): #both should be [1, 1, 1, 1024]   [0-1, 0-3, 0-5, 0-1]
        loss = []
        if (type(prediction)) is not list:
            loss = binarycrossentropy(target, prediction)
            return [loss]
        else:
            p1, p2, p3, p4 = prediction[0], prediction[1], prediction[2], prediction[3]#[...,0]
            t1, t2, t3, t4 = target[0][:,0], target[0][:,1], target[0][:,2], target[1]
            loss.append(self.w[0] * binarycrossentropy(t1[:,np.newaxis], p1))
            loss.append(self.w[1] * crossentropy(t2[:,np.newaxis], p2))
            loss.append(self.w[2] * crossentropy(t3[:,np.newaxis], p3))
            loss.append(self.w[3] * mean_squared(t4, p4))
            return loss

    def validation_loss(self, prediction, target):
        loss = []
        if (type(prediction)) is not list:
            loss = binarycrossentropy(target, prediction)
            return [loss]
        else:
            p1, p2, p3, p4 = prediction[0], prediction[1], prediction[2], prediction[3]#[...,0]
            t1, t2, t3, t4 = target[0][:, 0], target[0][:, 1], target[0][:, 2], target[1]
            loss.append(self.w[0] * binarycrossentropy(t1[:, np.newaxis], p1))
            loss.append(self.w[1] * crossentropy(t2[:, np.newaxis], p2))
            loss.append(self.w[2] * crossentropy(t3[:, np.newaxis], p3))
            loss.append(self.w[3] * mean_squared(t4, p4))
            return loss

    def gradient_penalty(self,gradient):
        grad_l2 = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)
        grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))
        return grad_penalty

### Step functions
    def train_stp(self, epoch, optimizer):
        data = next(self.train_data.next_batch(self.bs))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            with tf.GradientTape(persistent=True) as t:
                t.watch(self.model.trainable_variables)
                prediction = self.model(data[0], training=True) #
                self.latent_space = prediction[-1]
                self.labels = data[2]
                #print(self.labels)
                #print(np.amin(prediction[-2]), np.amax(prediction[-2]), np.mean(prediction[-2]))
                #print(np.amin(prediction[-1]), np.amax(prediction[-1]), np.mean(prediction[-1]))
                #print(np.amin(prediction[-2]), np.amax(prediction[-2]), np.mean(prediction[-2]))
                #print(np.amin(prediction[0]), np.amax(prediction[0]), np.mean(prediction[0]))
                prediction = prediction[:-1] # [:-1]!!!
                if self.model_type == 'autoencoder':
                    losses = self.training_loss(prediction, [data[2], data[0]])  ## labels, time_spectra
                elif self.model_type == 'automap':
                    losses = self.training_loss(prediction, data[2])
            opt = optimizer[0]
            # Gradient penalty
            #grads = t.gradient(losses, self.model.trainable_variables)
            grad_penalty = []
            #for g in grads:
            #    grad_penalty.append(self.gradient_penalty(g))
            #losses[1] += self.lambda_gp * np.asarray(grad_penalty)[0]
            #losses[1] += self.lambda_gp * np.asarray(grad_penalty)[1] # just penalize adverserial
            #losses[2] += self.lambda_gp * np.asarray(grad_penalty)[2]
            #losses = [a+b for a,b in zip(losses, self.lambda_gp*np.asarray(grad_penalty))]


        # Now update the gradient
        gradients = tape.gradient(losses, self.model.trainable_variables)
        opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return gradients, losses

    def validation_stp(self, epoch):
        data = next(self.val_data.next_batch(self.bs))
        prediction = self.model(data[0], training=False)
        if self.model_type == 'autoencoder':
            losses = self.validation_loss(prediction, [data[2], data[0]])
        elif self.model_type == 'automap':
            losses = self.validation_loss(prediction, data[2])
        return tf.reduce_sum(losses)

### Training function
   # @tf.function
    def train(self):
        if self.config['logger'] == 'tensorboard':
            train_log_dir = 'logs/gradient_tape/' + self.time + '/train'
            test_log_dir  = 'logs/gradient_tape/' + self.time + '/test'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

            grad_log_dir = 'logs/gradient_tape/' + self.time + '/grads'
            grad_summary_writer = tf.summary.create_file_writer(grad_log_dir)

            wb_log_dir = 'logs/gradient_tape/' + self.time + '/wb'
            wb_summary_writer = tf.summary.create_file_writer(wb_log_dir)

        elif self.config['logger'] == 'wandb':
            import wandb
            wandb.init(entity="mxm",
                       config=self.config)

        loss = np.zeros((2, self.eps))
        layer_names = [l.name for l in self.model.layers]
        opt = []
        for lr in self.lr:
            opt.append(tf.keras.optimizers.Adam(learning_rate=lr))
        for epoch in range(self.eps):
            print('epoch {} / {}'.format(epoch, self.eps))
            if epoch % 10 == 0:
                #print('epoch {} / {}'.format(epoch, self.eps))

                ###    EVALUATION STUFF   ###
                # Evaluate model:
                #train_time = np.load(join(self.path, 'train/train_time_data.npy'), allow_pickle=True)[:self.config['max_train_spectra']]
                #train_labels = np.load(join(self.path, 'train/train_labels.npy'), allow_pickle=True)[:self.config['max_train_spectra']]
                #print('Train set:')
                #pred = self.model.predict(train_time)
                #acc_train = accuracy(pred, train_labels)
                #print('False negative rate: {}'.format(false_neg_rate(train_labels[:,0], (pred[0] > 0.5)*1)))
                #self.pred_pid, self.pred_mid, self.pred_glioma = np.argmax(pred[1],axis=1), np.argmax(pred[2],axis=1), np.round(pred[0]).ravel()
                #acc_train, cm_pid_train, cm_mid_train = evaluate(pred, train_labels, plot_cm=True, name='CM_train_in_train', pid_cm=True, mid_cm=True, epoch=epoch)
                # Internal validation:
                #val_time = np.load(join(self.path, 'validate/validate_time_data.npy'), allow_pickle=True)[:None]
                #val_labels = np.load(join(self.path, 'validate/validate_labels.npy'), allow_pickle=True)[:None]
                #print('Validation set:')
                #pred = self.model.predict(val_time)
                #acc_val = accuracy(pred, val_labels)
                #print('False negative rate: {}'.format(false_neg_rate(val_labels[:,0], (pred[0] > 0.5)*1)))
                ### EXTERNAL VALIDATION ###
                '''val_time = np.load(join(self.path, 'nhx/test/test_time_data.npy'), allow_pickle=True)[:None]
                val_labels = np.load(join(self.path, 'nhx/test/test_labels.npy'), allow_pickle=True)[:None]
                print('External set NHX:')
                pred = self.model.predict(val_time)
                acc_nhx = accuracy(pred, val_labels)
                print('False negative rate: {}'.format(false_neg_rate(val_labels[:, 0], (pred[0] > 0.5) * 1)))
                val_time = np.load(join(self.path, 'ntnu/test/test_time_data.npy'), allow_pickle=True)[:None]
                val_labels = np.load(join(self.path, 'ntnu/test/test_labels.npy'), allow_pickle=True)[:None]
                print('External set NTNU:')
                pred = self.model.predict(val_time)
                acc_ntnu = accuracy(pred, val_labels)
                print('False negative rate: {}'.format(false_neg_rate(val_labels[:, 0], (pred[0] > 0.5) * 1)))'''

                #acc_val, _, _ = evaluate(pred, val_labels, plot_cm=False, name='CM_val_in_train', pid_cm=False, mid_cm=False, epoch=epoch)
                # Log t-SNE:
                #if epoch > 1:
                #    for i in range(3):
                #        tSNE(self.latent_space, self.labels[...,i], len(np.unique(self.labels[...,i])), folder='tsne_'+str(i),filename=str(epoch)+'.png')

            if (epoch % 10 == 0):# or (epoch % 10 == 1):
                ###    LOGGING STUFF   ###
                if self.config['logger'] == 'tensorboard':
                    with wb_summary_writer.as_default():
                        for i,l in enumerate(self.model.layers):
                            try:
                                weights = l.get_weights()[0]
                                tf.summary.histogram('weights_{}'.format(layer_names[i]), weights, step=epoch)
                            except:
                                continue
                            try:
                                biases = l.bias.numpy()
                                tf.summary.histogram('biases_{}'.format(layer_names[i]), biases, step=epoch)
                            except:
                                continue
                '''elif self.config['logger'] == 'wandb':
                    for i, l in enumerate(self.model.layers):
                        try:
                            weights = l.get_weights()[0]
                            wandb.log({'wb/weights': weights})
                        except:
                            continue
                        try:
                            biases = l.bias.numpy()
                            wandb.log({'wb/biases': biases})
                        except:
                            continue
                    wandb.log({
                        'CM/patient_ID': wandb.plot.confusion_matrix(preds=self.pred_mid, y_true=train_labels[:,2]),
                        'CM/machine_ID': wandb.plot.confusion_matrix(preds=self.pred_pid, y_true=train_labels[:,1]),
                        'CM/glioma' : wandb.plot.confusion_matrix(preds=self.pred_glioma, y_true=train_labels[:,0])
                    })'''

            ###    UPADATE STUFF   ###
            for step in range(self.num_spec//self.bs):
                train_grads, train_losses = self.train_stp(epoch,opt)
                train_loss = tf.reduce_sum(train_losses)
                val_loss = self.validation_stp(epoch)

            ###    LOGGING STUFF   ###
            if self.config['logger'] == 'tensorboard':
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss, step=epoch)
                    tf.summary.scalar('glioma_bce', train_losses[0], step=epoch)
                    tf.summary.scalar('pid_ce', train_losses[1], step=epoch)
                    tf.summary.scalar('mid_ce', train_losses[2], step=epoch)
                    tf.summary.scalar('spec_mse', train_losses[3], step=epoch)
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=epoch)
                with grad_summary_writer.as_default():
                    for i,g in enumerate(train_grads):
                        curr_grad = g
                        mean = tf.reduce_mean(tf.abs(curr_grad))
                        norm = tf.norm(curr_grad)
                        tf.summary.scalar('mean_grad_layer_{}'.format(i+1), mean, step=epoch)
                        tf.summary.scalar('norm_grad_layer_{}'.format(i+1), norm, step=epoch)
                        tf.summary.histogram('grad_histogram_layer_{}'.format(i+1), curr_grad, step=epoch)

            elif self.config['logger'] == 'wandb':
                i = 0
                wandb.log({
                    'loss/loss': train_loss,
                    'loss/glioma_bce': train_losses[0],
                    'loss/pid_ce': train_losses[1],
                    'loss/mid_ce': train_losses[2],
                    'loss/spec_mse': train_losses[3],
                    'loss/val_loss': val_loss,
                    #'acc/train_accuracy': acc_train,
                    #'acc/validation_accuracy': acc_val,
                    #'acc/external_accuracy_nhx': acc_nhx,
                    #'acc/external_accuracy_ntnu': acc_ntnu
                          })
                '''for i, g in enumerate(train_grads):
                    curr_grad = g
                    wandb.log({'norm_grads/norm_grad_layer{}'.format(i+1): tf.norm(curr_grad)})
                    wandb.log({'grads/grad_layer_{}'.format(i+1): curr_grad})'''

        ###    ENDING STUFF   ###
            loss[0,epoch], loss[1,epoch] = train_loss, val_loss

        tf.keras.utils.plot_model(self.model, to_file='{}.png'.format(self.name), show_shapes=True)
        return loss, self.model
