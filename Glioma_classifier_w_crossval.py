import numpy as np
from datetime import datetime
import os

from DataGenerator.FFT_data_generator import*
from DataGenerator.load_data import LoadData
from Models.Automap_w_classifier import classifier_1
#from Models.Autoencoder_v1 import *
#from Models.Autoencoder_v2 import *
from Models.Autoencoder_dense import *
#from Models.Autoencoder_lstm import *
#from Trainers.Class_trainer_v2 import Trainer
from Trainers.Class_trainer import Trainer

tf.config.experimental_run_functions_eagerly(True)
#tf.config.run_functions_eagerly(True)

from Utils.evaluation import evaluate
from Utils.get_args import get_args
from Utils.get_config import get_config


import warnings
warnings.filterwarnings("ignore")

def main():
    try:
        args = get_args()
        config = get_config(args.config)
        path = config['data_path']
    except:
        print('Missing arguments or could not read config file')
        exit(0)

    config['datetime'] = str(datetime.now())

    if config['model'] == 'autoencoder':  # labels: 0=glioma, 1=PID, 2=MID
        config['labels'] = [0, 1, 2]
    else:
        config['labels'] = [0]

    LD = LoadData(config)
    LD.create_dataset(types=['gbm', 'healthy'])

    for k in range(config['cross_validation']):
        # Data loader
        LD.train_test_split(k, types=['gbm', 'healthy'])
        train_data = DataGenerator(N=config['max_train_spectra'], use_labels=config['labels'])
        val_data   = ValGenerator(N=config['max_val_spectra'], use_labels=config['labels'])
        # Set up model
        dim = train_data.dim   #train_data[0].shape # N*t*2 (N,1024,2)
        if config['model'] == 'automap':
            model = classifier_1(dim[1:], 1, config)
        elif config['model'] == 'autoencoder':
            model = autoencodeclass(dim[1:], config)

        trainer = Trainer(model, train_data, val_data, config)
        loss, model = trainer.train()

        if config['save_model']:
            folder_name = 'kfold_models/'
            try:
                save_name = folder_name + config['run_name'] + '_{}_'.format(k) + config['datetime']
                model.save('{}'.format(save_name))
                print('Model with weights saved as: {}'.format(save_name))
            except ValueError:
                save_name = folder_name + 'saved_weights' + '_{}_'.format(k) + config['datetime'] + '.h5'
                model.save_weights(save_name)
                print('Weights saved as: {}'.format(save_name))

            np.save(config['run_name']+'_class_loss.npy', loss)

    #path_ = os.path.join(path, 'gbm')
    #labtest = np.load(os.path.join(path, 'train/train_time_data.npy'), allow_pickle=True)[150][np.newaxis,...]
    #np.save('test_reconstruction.npy', model.predict(labtest)[-2])
    #np.save('input_reconstruction.npy', labtest)

    ### Validate model
    N1, N2 = config['max_train_spectra'], None
    print('##########################################################################')
    print('Final evaluation:')
    # Internal validation (subset)
    #train_time = np.load(join(config['data_path'], 'train/train_time_data.npy'), allow_pickle=True)[:N1]
    #train_labels = np.load(join(config['data_path'], 'train/train_labels.npy'), allow_pickle=True)[:N1]

    #val_time = np.load(join(config['data_path'], 'validate/validate_time_data.npy'), allow_pickle=True)[:N2]
    #val_labels = np.load(join(config['data_path'], 'validate/validate_labels.npy'), allow_pickle=True)[:N2]

    test_time = np.load(join(config['data_path'], 'gbm/train/train_time_data.npy'), allow_pickle=True)
    #test_time = np.load(join(config['data_path'], 'tumor/train_time_data.npy'), allow_pickle=True)
    result = (model.predict(test_time)[0] > 0.5) * 1
    print('Just glioma accuracy:', np.sum(result) / len(result))

    test_time = np.load(join(config['data_path'], 'healthy/train/train_time_data.npy'), allow_pickle=True)
    #test_time = np.load(join(config['data_path'], 'healthy/train_time_data.npy'), allow_pickle=True)
    result = (model.predict(test_time)[0] > 0.5) * 1
    print('Just healthy accuracy:', np.sum(result) / len(result))

    if False:
        print('Training data:')
        pred = model.predict(test_time)
        #pred = model.predict(train_time)
        evaluate(pred, train_labels, plot_cm=True, name='CM_training', save_pred=True)
        print('Validation data:')
        pred = model.predict(val_time)
        evaluate(pred, val_labels, plot_cm=True, name='CM_validation', save_pred=True)

    if config['external_val']:
        # External validation (new set)
        test_time = np.load(join(config['data_path'], 'test/test_time_data.npy'), allow_pickle=True)[:N2]
        test_labels = np.load(join(config['data_path'], 'test/test_labels.npy'), allow_pickle=True)[:N2]

        print('External data:')
        pred = model.predict(test_time)
        evaluate(pred, test_labels, plot_cm=True, name='CM_test_nhx')

        test_time = np.load(join(config['data_path'], 'test_gbm/test_time_data.npy'), allow_pickle=True)
        # test_time = np.load(join(config))
        result = (model.predict(test_time)[0] > 0.5) * 1
        print('EXT  Just glioma accuracy:', np.sum(result) / len(result))

        test_time = np.load(join(config['data_path'], 'test_healthy/test_time_data.npy'), allow_pickle=True)
        result = (model.predict(test_time)[0] > 0.5) * 1
        print('EXT  Just healthy accuracy:', np.sum(result) / len(result))


if __name__ == '__main__':
    main()