import numpy as np
import matplotlib.pyplot as plt

class Visualise:

    def __init__(self, target, prediction, history, threshold=0.2):
        self.t = prediction.shape[0]
        self.targ = target[0]#[:self.t,:]
        self.pred = prediction[0]
        self.x = np.arange(self.t)
        self.history = history
        self.thr = threshold
        self.x = 0

    def performance(self):
        error = []
        for i in range(self.t):
            error.append(self.mse(i))
        error = np.array(error)
        mse_above = len(np.where(error > self.thr)[0])
        under_thr = (1 - mse_above / self.t) * 100
        print('Bellow threshold (acceptable): {}%'.format(under_thr))
        return

    def mse(self, idx=0):
        err = np.mean(np.square(self.pred - self.targ))
        # complex to normal...
        return err.real + err.imag

    def error(self, N=16, ymin=None, ymax=None, cols=4):
        col = lambda x: 'g' if (self.mse(x) <= self.thr) else 'r'
        fig, axs = plt.subplots(1) #int(np.ceil(N / cols)), cols, figsize=(30, 15))
        y1, y2 = self.targ, self.pred #self.targ[ymin:ymax], self.pred[ymin:ymax]
        # x = range(len(self.pred[0]))[ymin:ymax]
        x = self.x #range(len(y1))
        axs.plot(y1, label='Target', c='b')
        axs.plot(y2, label='Prediction', c=col(0), alpha=0.75)
        axs.fill_between(x, y1[np.newaxis], y2[np.newaxis], label='Error', color=col(0), alpha=0.5)
        axs.legend()
        plt.setp(axs, ylim=[np.amin([y1,y2])*1.1, np.amax([y1,y2])*1.1])
        fig.suptitle('Model Prediction', fontsize=20)

    def histogram_class(self):
        title = 'Histogram of Mean Square Error'
        labels = 'Normal', 'Abnormal'
        return

    def histogram_mse(self, bins=10):
        error = []
        for i in range(self.t):
            error.append(self.mse(i))
        error = np.array(error)
        error = error[np.where(error > self.thr)[0]]
        plt.hist(error,bins=bins)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.title('Histogram of MSE')
        return


    def ROC(self):
        # Plot true positive rate vs false positive rate
        return