import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from os.path import join

def CM(target, pred, glioma=True, save_plot=False, filename='', folder=''):
    target = target.astype('float32')
    n_classes = int(np.amax(target)+1)
    cm = tf.math.confusion_matrix(target, pred)
    if save_plot:
        cm_norm = cm / len(pred)
        threshold = int(len(pred) * 0.2)
        fig = plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        if glioma:
            tick_marks = np.arange(n_classes)
            classes = ['Healthy', 'Glioma']
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
        for i, j in itertools.product(range(n_classes), range(n_classes)):
            plt.text(j,i, f'{cm[i,j]}' , horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black')
           # plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)', horizontalalignment='center',
           #          color='white' if cm[i, j] > threshold else 'black')
        plt.savefig(join(folder, filename))
        plt.clf()
        plt.cla()
        plt.close()
    return cm