import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import math
import sklearn
import tensorflow as tf
from tensorflow.keras import callbacks



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



class ShowProgress(callbacks.Callback):
    def __init__(self, epochs, step_show=1, metric="accuracy"):
        super(ShowProgress, self).__init__()
        self.epochs = epochs
        self.step_show = step_show
        self.metric = metric

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(range(self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step_show == 0:

            self.pbar.set_description(f"""Epoch : {epoch + 1} / {self.epochs}, 
            Train {self.metric} : {round(logs[self.metric], 4)}, 
            Valid {self.metric} : {round(logs['val_' + self.metric], 4)}""")

            self.pbar.update(self.step_show)

            
class BestModelWeights(callbacks.Callback):
    def __init__(self, metric="val_accuracy", metric_type="max"):
        super(BestModelWeights, self).__init__()
        self.metric = metric
        self.metric_type = metric_type
        if self.metric_type not in ["min", "max"]:
                raise NameError('metric_type must be min or max')

    def on_train_begin(self, logs=None):
        if self.metric_type == "min":
            self.best_metric = math.inf
        else:
            self.best_metric = -math.inf
        self.best_epoch = 0
        self.model_best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if self.metric_type == "min":
            if self.best_metric >= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch
        else:
            if self.best_metric <= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_best_weights)
        print(f"\nBest weights is set, Best Epoch was : {self.best_epoch+1}\n")


def plot_metrics(history):
    plt.figure(figsize=(12, 10))
    metrics = ['loss', 'prc', 'accuracy', 'fp', 'precision', "tp", "recall", "tn", "auc", "fn"]
    
    for n, metric in enumerate(metrics):
        
        name = metric.replace("_"," ").capitalize()
        plt.subplot(5, 2, n+1)
        
        plt.plot(history.epoch,
                 history.history[metric],
                 color=colors[0],
                 label='Train')
        
        plt.plot(history.epoch,
                 history.history['val_'+ metric],
                 color=colors[1],
                 #linestyle="--",
                 label='Val')
        
        plt.xlabel('Epoch')
        plt.ylabel(name)

        plt.legend();


def plot_roc(name, labels, predictions, **kwargs):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=name + f" ( AUC = {round(auc, 3)} )", linewidth=2, **kwargs)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.grid(True)


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()


def plot_cm(name, labels, predictions, p=0.5):
    cm = sklearn.metrics.confusion_matrix(labels, predictions > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(name + ' Confusion matrix @ {:.2f}'.format(p))
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')