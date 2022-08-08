import tensorflow as tf
import numpy as np

from Utils.confusion_matrix import CM
from Utils.roc_curve import ROC
from Utils.FFT_losses import false_neg_rate
#from sklearn import # one hot ...

def accuracy(pred, target):
    pred_0 = (pred[0] > 0.5) * 1
    m = tf.keras.metrics.Accuracy()
    acc = m(target[:, 0].astype('float32'), pred_0).numpy()
    print('Accuracy: {}'.format(acc))
    print('Predicted GBM {}% \n Actual GBM {} %'.format(np.sum(pred_0) / len(pred_0) * 100,
                                                        np.sum(target[:, 0].astype('float32')) / len(target[:, 0].astype('float32')) * 100))
    return acc

def evaluate(pred, target, plot_cm=True, plot_roc=True, name='test_fig', pid_cm=False, mid_cm=False, epoch=0, save_pred=False):
    if save_pred:
        np.save('pred_'+name, pred[0])
    #pred_0 = (pred[0] > 0.5) * 1

    # Accuracy:
    acc = accuracy(pred, target)
    cm = CM(target[:, 0].astype('float32'), (pred[0] > 0.5)*1, save_plot=plot_cm, filename=name, folder='evaluation')

    print('False negative rate: {}'.format(false_neg_rate(target[:,0], (pred[0] > 0.5)*1)))

    ROC(target[:, 0].astype('float32'), pred[0], save_plot=plot_roc, filename='ROC'+name, folder='evaluation')

    cm_pid, cm_mid = None, None
    if pid_cm:
        # Confusion matrix for patient-ID:
        pred_1 = np.argmax(pred[1], axis=1)
        target1 = target[:,1].astype('float32') # -> to one hot?
        cm_mid = CM(target1, pred_1, glioma=False, save_plot=True, filename='CM_' + str(epoch), folder='CM_PID')

    if mid_cm:
        pred_2 = np.argmax(pred[2], axis=1)
        target2 = target[:, 2].astype('float32')  # -> to one hot?
        CM(target2, pred_2, glioma=False, save_plot=True, filename='CM_' + str(epoch), folder='CM_MID')

    return(acc, cm_pid, cm_mid)