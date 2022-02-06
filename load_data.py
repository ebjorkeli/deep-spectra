"Load MR Spectroscopy data for training domain transform (Fourier Transform)"

import numpy as np

import os.path
from os import path
from os.path import join
from pathlib import Path
home = str(Path.home())

import scipy.io as sio

class LoadData:

    def __init__(self):
        #self.df = join(home, 'Documents', 'phd',)
        self.df = join('/Users', 'erinbjorkeli', 'phd',)

    def load_data_phantom(self, M=7):
        data = []
        # Load dataset 1:
        for i in range(1, M+1):
            file = 'Spec_{}.mat'.format(i)
            path_to_file = join(self.df, 'Data_Spectra ', file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_['Spec_{}'.format(i)])
        # Load dataset 2:
        path_to_file = join(self.df, 'Data_Spectra ', 'Spec_1801.mat')
        d_ = sio.loadmat(path_to_file)
        data.append(d_['Spec'])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x,y

    def load_data_phase_adj(self):
        data = []
        path_to_file = join(self.df, 'Data_Spectra ', '1aAHSPEC.mat')
        data.append(sio.loadmat(path_to_file)['MRSI_2d'])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x,y

    def load_data_patients(self, type='all'):
        data, keys= [], []
        if type == 'all':
            files = ['Spec_GBM_7939_1024.mat', 'Spec_Healthy_volunteer_11251_1024.mat', 'Healthy_subj018_3M.mat',
                     'Healthy_subj0137_3M.mat', 'Healthy_subj199_3M.mat', 'Healthy_subj018_5M.mat',
                     'Healthy_subj0137_5M.mat', 'Healthy_subj199_5M.mat']
            keys += ['Spec']*2 + ['MRSI_2d']*6
        elif type == 'gbm':
            # Load just glioblastoma
            files = ['Spec_GBM_7939_1024.mat']
            keys += ['Spec']
        elif type == 'healthy':
            files = ['Spec_Healthy_volunteer_11251_1024.mat', 'Healthy_subj018_3M.mat',
                     'Healthy_subj0137_3M.mat', 'Healthy_subj199_3M.mat', 'Healthy_subj018_5M.mat',
                     'Healthy_subj0137_5M.mat', 'Healthy_subj199_5M.mat']
            keys += ['Spec'] + ['MRSI_2d']*6
        for i, file in enumerate(files):
            path_to_file = join(self.df, 'Data_Spectra ', file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_[keys[i]])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x,y

    def load_all(self):
        data, files, keys = [], [], []
        for i in range(1,M+1):
            files.append('Spec_{}.mat'.format(i))
            keys.append('Spec_{}'.format(i))
        files += ['Spec_1801.mat', 'Spec_GBM_7939_1024.mat', 'Spec_Healthy_volunteer_11251_1024.mat', '1aAHSPEC.mat',
                  'Healthy_subj018_3M.mat', 'Healthy_subj0137_3M.mat', 'Healthy_subj199_3M.mat',
                  'Healthy_subj018_5M.mat','Healthy_subj0137_5M.mat', 'Healthy_subj199_5M.mat']
        keys += ['Spec']*3 + ['MRSI_2d']*7

        for i, file in enumerate(files):
            path_to_file = join(self.df, 'Data_Spectra ', file)
            d_ = sio.loadmat(path_to_file)
            data.append(d_[keys[i]])

        data = np.concatenate(data)
        x, y = data[:,:,:2], data[:,:,2:]
        return x, y