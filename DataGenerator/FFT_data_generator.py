import numpy as np
from os.path import join
path = '/Users/erinbjorkeli/phd/Data_Spectra /ReIm_data_3' # currently here, move data later...
#path = 'Data/example_spectra'
#path = '/Users/erinbjorkeli/phd/Data_Spectra /temp_data'

path_h = join(path, 'healthy')
path_g = join(path, 'gbm')

# Currently the labels are 0=glioma, 1=patientID, 2=machineID, use_labels can be used to select them

def get_batch(batch_size, i, iterations):
    if i + batch_size >= len(iterations):
        rest = (i + batch_size) % len(iterations)
        idx = np.concatenate((iterations[i:], iterations[:rest]), axis=0)
        i = rest
    else:
        idx = iterations[i:i + batch_size]
        i += batch_size
    return idx, i

cat = np.concatenate

class DataGenerator:
    def __init__(self, N=None, use_labels=None, white_noise=None):
        self.use_labels = use_labels
        if N is not None:
            N = int(np.ceil(N/2))
        self.x1, self.y1 = np.load(join(path_h, 'train/train_time_data.npy'))[:N], \
                         np.load(join(path_h, 'train/train_freq_data.npy'))[:N]
        self.x2, self.y2 = np.load(join(path_g, 'train/train_time_data.npy'))[:N], \
                         np.load(join(path_g, 'train/train_freq_data.npy'))[:N]

        if white_noise is not None:
            mean = 0
            std = white_noise
            self.x1 += np.random.normal(mean, std, size=self.x1.shape)
            self.x2 += np.random.normal(mean, std, size=self.x2.shape)

        if self.use_labels is not None:
            self.labels1 = np.load(join(path_h, 'train/train_labels.npy'), allow_pickle=True)[:N][:,use_labels]
            self.labels2 = np.load(join(path_g, 'train/train_labels.npy'), allow_pickle=True)[:N][:, use_labels]

        self.dim1, self.dim2 = self.x1.shape, self.x2.shape
        self.dim = [self.dim1[0], self.dim1[1], self.dim1[2]]
        self.dim[0] *= 2
        self.iterations1, self.iterations2 = np.arange(self.dim1[0]), np.arange(self.dim2[0])
        np.random.shuffle(self.iterations1)   # can set a seed, but then average over performance...
        np.random.shuffle(self.iterations2)
        self.i = [0,0]


    def next_batch(self, batch_size):
        batch_size = batch_size //2
        idx1, self.i[0] = get_batch(batch_size, self.i[0], self.iterations1)
        idx2, self.i[1] = get_batch(batch_size, self.i[1], self.iterations2)
        if self.use_labels is not None:
            #print(self.labels1[idx1], self.labels2[idx2])
            yield cat((self.x1[idx1], self.x2[idx2]), axis=0),\
                  cat((self.y1[idx1], self.y2[idx2]), axis=0),\
                  cat((self.labels1[idx1], self.labels2[idx2]), axis=0)
            #yield self.x[idx], self.y[idx], self.labels[idx]
        else:
            #yield self.x[idx], self.y[idx], None
            yield cat((self.x1[idx1], self.x2[idx2]), axis=0),\
                  cat((self.y1[idx1], self.y2[idx2]), axis=0)


class ValGenerator:
    def __init__(self, N=None, use_labels=None):
        self.use_labels = use_labels
        if N is not None:
            N = int(np.ceil(N/2))
        self.x = cat((np.load(join(path_h, 'validate/validate_time_data.npy'))[:N],
                     np.load(join(path_g, 'validate/validate_time_data.npy'))[:N]), axis=0)
        self.y = cat((np.load(join(path_h, 'validate/validate_freq_data.npy'))[:N],
                     np.load(join(path_g, 'validate/validate_freq_data.npy'))[:N]), axis=0)
        if self.use_labels is not None:
            self.labels = cat((np.load(join(path_h, 'validate/validate_labels.npy'), allow_pickle=True)[:N][:,use_labels],
                              np.load(join(path_g, 'validate/validate_labels.npy'), allow_pickle=True)[:N][:,use_labels]), axis=0)
        self.dim = self.x.shape
        self.i = 0
        self.iterations = np.arange(self.dim[0])
        np.random.shuffle(self.iterations)

    def next_batch(self, batch_size):
        batch_size = batch_size // 2
        idx, self.i = get_batch(batch_size, self.i, self.iterations)
        if self.use_labels is not None:
            yield self.x[idx], self.y[idx], self.labels[idx]
        else:
            yield self.x[idx] ,self.y[idx]

class TestGenerator:
    def __init__(self, N=None, use_labels=None):
        self.use_labels = use_labels
        self.x, self.y = np.load(join(path, 'test/test_time_data.npy'))[:N],\
                         np.load(join(path, 'test/test_freq_data.npy'))[:N]
        if self.use_labels is not None:
            self.labels = np.load(join(path, 'test/test_labels.npy'), allow_pickle=True)[:N][:,use_labels]
        self.dim = self.x.shape
        self.i = 0

    def next_batch(self, batch_size):
        idx = np.arange(self.i,self.i+batch_size)
        self.i += batch_size
        if self.use_labels is not None:
            yield self.x[idx], self.y[idx], self.labels[idx]
        else:
            yield self.x[idx], self.y[idx], None