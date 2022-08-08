from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from os.path import join

def ROC(target, prediction, save_plot=True, filename='', folder=''):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    AUC = auc(fpr, tpr)

    if save_plot:
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(AUC))
        plt.xlabel('1 - Specifisity (FPR)')
        plt.ylabel('Sensitivity (TPR)')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(join(folder, filename))
        plt.clf()
        plt.cla()
        plt.close()
    return (fpr, tpr, AUC)