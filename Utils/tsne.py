from os.path import join
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def tSNE(X, y, n_labels, folder, filename):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright", n_labels)

    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(X, y)
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)

    plt.savefig(join(folder, filename))
    plt.clf()
    plt.cla()
    plt.close()