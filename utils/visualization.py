import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cfg, y_true, y_pred, epoch, set_type):
    """
    Plots and saves confusion matrix at given epoch.
    :param cfg: train config
    :param y_true: labels
    :param y_pred: predictions
    :param epoch: epoch to save at
    :param set_type: 'train' or 'test' data type
    """
    conf_m = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(conf_m, annot=True, fmt='.0f', cbar=False, annot_kws={'size': 16})
    ax.set_yticklabels(np.arange(conf_m.shape[0]), fontsize=14, rotation=0)
    ax.set_xticklabels(np.arange(conf_m.shape[0]), fontsize=14, rotation=0)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.savefig(cfg.eval_plots_dir + f'/confusion_matrix_epoch_{epoch}_set_type_{set_type}.png')
    plt.cla()
    plt.close()
