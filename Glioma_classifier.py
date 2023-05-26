import numpy as np
from datetime import datetime
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt


from DataGenerator.FFT_data_generator2 import *
#from Models.Automap_w_classifier import classifier_1
from Models.DenseNet_class import *

#tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)

from Utils.evaluation import evaluate
from Utils.get_args import get_args
from Utils.get_config import get_config


import warnings
warnings.filterwarnings("ignore")

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=0):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def main():
    try:
        args = get_args()
        config = get_config(args.config)
        path = config['data_path']
    except:
        print('Missing arguments or could not read config file')
        exit(0)

    set_global_determinism(seed=config['seed'])
    config['datetime'] = str(datetime.now())

    # Data loader
    if config['model'] == 'autoencoder': # labels: 0=glioma, 1=PID, 2=MID, 3=IDH
        config['labels'] = [0,1,2]
    elif config['model'] == 'densenet':
        config['labels'] = [0,3]
    else:
        config['labels'] = [0]

    #config['labels'] = None

    train_data = DataGenerator(config['batch_size'], N=config['max_train_spectra'], use_labels=config['labels'])
    val_data   = ValGenerator(config['batch_size'], N=config['max_val_spectra'], use_labels=config['labels'])
    # Set up model
    dim = train_data.dim   #train_data[0].shape # N*t*2 (N,1024,2)
    if config['model'] == 'automap':
        model = classifier_1(dim[1:], 1, config)
    elif config['model'] == 'autoencoder':
        model = autoencodeclass(dim[1:], config)
    elif config['model'] == 'densenet':
        model = densenet(dim[1:], [1024,1], f=4)

    #trainer = Trainer(model, train_data, val_data, config)

    # Alternative training (for densenet++)
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(opt,
                  loss={#'domain_output' : tf.keras.losses.MeanSquaredError(),})
                        'glioma_output' : tf.keras.losses.BinaryCrossentropy(),
                        'idhmut_output' : tf.keras.losses.BinaryCrossentropy()},
                  metrics={'glioma_output' : 'accuracy',
                           'idhmut_output' : 'accuracy'})
    history = model.fit(x=train_data,
                        steps_per_epoch=dim[0] // config['batch_size'],
                        epochs=config['max_epochs'],
                        validation_data=val_data,
                        validation_steps=val_data.dim[0] // config['batch_size'])

    if config['save_model']:
        try:
            save_name = config['run_name'] + '_' + config['datetime']
            model.save('{}'.format(save_name))
            print('Model with weights saved as: {}'.format(save_name))
        except ValueError:
            save_name = 'saved_weights' + '_' + config['datetime'] + '.h5'
            model.save_weights(save_name)
            print('Weights saved as: {}'.format(save_name))

    #plt.plot(history.history['loss'])
    #plt.show()
    plt.plot(history.history['glioma_output_loss'])
    plt.show()
    plt.plot(history.history['idhmut_output_loss'])
    plt.show()

    np.save(config['run_name']+'_class_history.npy', history.history["loss"])


if __name__ == '__main__':
    main()