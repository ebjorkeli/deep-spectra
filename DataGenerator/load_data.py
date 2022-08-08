"Load MR Spectroscopy data for training domain transform (Fourier Transform)"

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle

import os.path
from os import path
from os.path import join
from pathlib import Path
home = str(Path.home())

import scipy.io as sio

class LoadData:

    def __init__(self, config):
        self.path = '/Users/erinbjorkeli/phd/Data_Spectra '
        self.save_path = join(self.path, 'temp_data')
        self.k = config['cross_validation']

    # Load different dataset

    def load_data_phantom(self, M=7):
        data = []
        # Load dataset 1:
        for i in range(1, M+1):
            file = 'Spec_{}.mat'.format(i)
            path_to_file = join(self.path, file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_['Spec_{}'.format(i)])
        # Load dataset 2:
        path_to_file = join(self.path, 'Spec_1801.mat')
        d_ = sio.loadmat(path_to_file)
        data.append(d_['Spec'])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x,y

    def load_data_phase_adj(self):
        data = []
        path_to_file = join(self.path, '1aAHSPEC.mat')
        data.append(sio.loadmat(path_to_file)['MRSI_2d'])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x,y

    def load_all(self, type='all', extra_gbm=False):
        data, keys= [], []
        if type == 'all':
            files = ['Spec_GBM_7939_1024.mat', 'Spec_Healthy_volunteer_11251_1024.mat', 'Healthy_subj018_3M.mat',
                     'Healthy_subj0137_3M.mat', 'Healthy_subj199_3M.mat', 'Healthy_subj018_5M.mat',
                     'Healthy_subj0137_5M.mat', 'Healthy_subj199_5M.mat', 'MRSI_2D_FACBC_glioma_002_33.mat',
                     'MRSI_2D_FACBC_glioma_002_37.mat', 'MRSI_2D_FACBC_glioma_004_24.mat', 'MRSI_2D_FACBC_glioma_005_21.mat',
                     'MRSI_2D_FACBC_glioma_007_22.mat', 'MRSI_2D_FACBC_glioma_007_23.mat', 'MRSI_2D_FACBC_glioma_008_22.mat',
                     'MRSI_2D_FACBC_glioma_009_22.mat', 'MRSI_2D_FACBC_glioma_009_23.mat', 'MRSI_2D_FACBC_glioma_010_22.mat',
                     'MRSI_2D_FACBC_glioma_010_23.mat', 'MRSI_2D_FACBC_glioma_011_39.mat']
            keys += ['Spec']*2 + ['MRSI_2d']*18
        elif type == 'gbm':
            # Load just glioblastoma
            files = ['Spec_GBM_7939_1024.mat'] + extra_gbm * ['MRSI_2D_FACBC_glioma_002_33.mat',
                     'MRSI_2D_FACBC_glioma_002_37.mat', 'MRSI_2D_FACBC_glioma_004_24.mat', 'MRSI_2D_FACBC_glioma_005_21.mat',
                     'MRSI_2D_FACBC_glioma_007_22.mat', 'MRSI_2D_FACBC_glioma_007_23.mat', 'MRSI_2D_FACBC_glioma_008_22.mat',
                     'MRSI_2D_FACBC_glioma_009_22.mat', 'MRSI_2D_FACBC_glioma_009_23.mat', 'MRSI_2D_FACBC_glioma_010_22.mat',
                     'MRSI_2D_FACBC_glioma_010_23.mat', 'MRSI_2D_FACBC_glioma_011_39.mat']
            keys += ['Spec'] + ['MRSI_2d'] * 12 * extra_gbm
        elif type == 'healthy':
            files = ['Spec_Healthy_volunteer_11251_1024.mat', 'Healthy_subj018_3M.mat',
                     'Healthy_subj0137_3M.mat', 'Healthy_subj199_3M.mat', 'Healthy_subj018_5M.mat',
                     'Healthy_subj0137_5M.mat', 'Healthy_subj199_5M.mat']
            keys += ['Spec'] + ['MRSI_2d']*6
        for i, file in enumerate(files):
            path_to_file = join(self.path, file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_[keys[i]])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x,y


    def load_ntnu(self):
        data, files, keys = [], [], []
        files = ['MRSI_2D_FACBC_glioma_002_33.mat','MRSI_2D_FACBC_glioma_002_37.mat', 'MRSI_2D_FACBC_glioma_004_24.mat',
                 'MRSI_2D_FACBC_glioma_005_21.mat','MRSI_2D_FACBC_glioma_007_22.mat', 'MRSI_2D_FACBC_glioma_007_23.mat',
                 'MRSI_2D_FACBC_glioma_008_22.mat','MRSI_2D_FACBC_glioma_009_22.mat', 'MRSI_2D_FACBC_glioma_009_23.mat',
                 'MRSI_2D_FACBC_glioma_010_22.mat', 'MRSI_2D_FACBC_glioma_010_23.mat', 'MRSI_2D_FACBC_glioma_011_39.mat']
        keys += ['MRSI_2d']*12

        for i, file in enumerate(files):
            path_to_file = join(self.path, file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_[keys[i]])
        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x, y

    def load_nhx(self):
        sub_folder = 'NHX spectra Glioma'
        data, files, keys = [], [], []
        files = ['MRSI_2D_NHX_glioma_01.mat', 'MRSI_2D_NHX_glioma_03.mat', 'MRSI_2D_NHX_glioma_032.mat', \
                'MRSI_2D_NHX_glioma_033.mat', 'MRSI_2D_NHX_glioma_034.mat', 'MRSI_2D_NHX_glioma_036.mat', \
                'MRSI_2D_NHX_glioma_037.mat', 'MRSI_2D_NHX_glioma_039.mat', 'MRSI_2D_NHX_glioma_041.mat', \
                'MRSI_2D_NHX_glioma_042.mat', 'MRSI_2D_NHX_glioma_043.mat', 'MRSI_2D_NHX_glioma_044.mat', \
                'MRSI_2D_NHX_glioma_045.mat', 'MRSI_2D_NHX_glioma_046.mat']
        keys += ['MRSI_2d'] * 14
        for i, file in enumerate(files):
            path_to_file = join(self.path, sub_folder, file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_[keys[i]])
        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x, y


    # Utils:
    def norm(self, x):
        x_min = x.min(axis=(1), keepdims=True)
        x_max = x.max(axis=(1), keepdims=True)
        x = (x - x_min) / (x_max - x_min)
        x = np.nan_to_num(x, nan=0)
        return x

    def get_labels(self, query):
        df = pd.read_csv(join(self.path, 'labels_dataspectra2_extended.csv'))
        LE = preprocessing.LabelEncoder()
        le = LE.fit(df['PID'])

        df =  df.query(query)

        df = [df['Glioma'], LE.transform(df['PID']), df['MID']]
        df = np.concatenate(df)
        df = df.reshape(3, -1).T

        return df

    # Full loader stuff
    def load(self, type):
        if type == 'ntnu':
            x, y = self.load_ntnu()
            labels = self.get_labels(query='PID == "G2"')

        elif type == 'nhx':
            x, y = self.load_nhx()
            labels = self.get_labels(query='PID == "G3"')

        elif type == 'open':
            x, y = self.load_all()
            labels = self.get_labels()[:N]

        elif type == 'gbm':
            x, y = self.load_all('gbm')
            labels = self.get_labels(query='PID == "G1"')

        elif type == 'healthy':
            x, y = self.load_all('healthy')
            labels = self.get_labels(query='Glioma == 0')

        else:
            print('{} is not a valid type'.format(type))
            pass

        self.x, self.y, self.lab = self.norm(x), self.norm(y), labels
        self.N = self.x.shape[0]

    def k_split(self, split, seed=None):
        if seed is not None:
            random.seed(seed)
        x, y, lab = shuffle(self.x, self.y, self.lab, )

        div = int(self.N*split)
        xs, ys, labs = [], [], []
        for i in range(int(1/split)):
            xs.append(x[i*div:(i+1)*div])
            ys.append(y[i*div:(i+1)*div])
            labs.append(lab[i*div:(i+1)*div])

        return xs, ys, labs
        #train, test = [x_train, y_train, lab_train], [x_test, y_test, lab_test]
        #return train, test

    def train_test_split(self, k,types):

        # load train from 4, test from 1
        j  = 0
        for type in types:
            for i in range(self.k):
                if i == k:
                    x_test = np.load(join(self.save_path, type, str(i),'train_time_data.npy'))
                    y_test = np.load(join(self.save_path, type, str(i), 'train_freq_data.npy'))
                    lab_test = np.load(join(self.save_path, type, str(i), 'train_labels.npy'))
                else:
                    if j == 0:
                        x_train = np.load(join(self.save_path, type, str(i), 'train_time_data.npy'))
                        y_train = np.load(join(self.save_path, type, str(i), 'train_freq_data.npy'))
                        lab_train = np.load(join(self.save_path, type, str(i), 'train_labels.npy'))
                        j += 1
                    else:
                        x = np.load(join(self.save_path, type, str(i),'train_time_data.npy'))
                        y = np.load(join(self.save_path, type, str(i), 'train_freq_data.npy'))
                        lab = np.load(join(self.save_path, type, str(i), 'train_labels.npy'))

                        x_train = np.concatenate((x_train, x))
                        y_train = np.concatenate((y_train, y))
                        lab_train =  np.concatenate((lab_train, lab))

            train, test = [x_train, y_train, lab_train], [x_test, y_test, lab_test]
            np.save(join(self.save_path, type, 'train/train_time_data.npy'), train[0])
            np.save(join(self.save_path, type, 'train/train_freq_data.npy'), train[1])
            np.save(join(self.save_path, type, 'train/train_labels.npy'), train[2])

            np.save(join(self.save_path, type, 'validate/validate_time_data.npy'), test[0])
            np.save(join(self.save_path, type, 'validate/validate_freq_data.npy'), test[1])
            np.save(join(self.save_path, type, 'validate/validate_labels.npy'), test[2])
        return


    def create_dataset(self, types, split=0.2, seed=None):
        for type in types:
            self.load(type)
            xs,ys,labs = self.k_split(split, seed)
            for i in range(self.k):
                np.save(join(self.save_path, type, str(i),'train_time_data.npy'), xs[i])
                np.save(join(self.save_path, type, str(i), 'train_freq_data.npy'), ys[i])
                np.save(join(self.save_path, type, str(i), 'train_labels.npy'), labs[i])
        return

    # should be loaded where data is loaded first time, then this can be done repeatedly for cv.
    # read and check if true...