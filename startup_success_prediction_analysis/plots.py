from matplotlib import pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np


def plot_countplot(df, x, hue, x_label, y_label, figsize, sort_by_count, legend_labels=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    if (sort_by_count):
        cp = sns.countplot(x=x, hue=hue, data=df, palette="nipy_spectral",
                      order=df[x].value_counts().index)
    else:
        cp = sns.countplot(x=x, hue=hue, data=df, palette="nipy_spectral")
    
    cp.tick_params(axis='both', labelsize=12)
    cp.set_xticklabels(cp.get_xticklabels(), rotation=90)
    cp.set_xlabel(xlabel=x_label, fontsize=14)
    cp.set_ylabel(ylabel=y_label, fontsize=14)
    
    if legend_labels is None:
        ax.legend(loc='upper right', fontsize=13)
    else:
        ax.legend(labels=legend_labels, loc='upper right', fontsize=13)
    # plt.savefig('image')


def plot_corr_matrix(df_corr_matrix, fontsize, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(df_corr_matrix, dtype=np.bool))
    mask = mask[1:, :-1]
    df_corr_matrix = df_corr_matrix.iloc[1:,:-1]
    sns.heatmap(df_corr_matrix, annot=True, mask=mask, fmt='.2f', annot_kws={'size': fontsize})


def plot_confusion_matrix(confusion_matrix, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    cm = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='BuPu')
    cm.set_xlabel(xlabel='Predicted class', fontsize=14)
    cm.set_ylabel(ylabel='Actual class', fontsize=14)
    plt.title('Confusion matrix')


def plot_precision_recall_curve(classifier, X_test, y_test, label, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    probas = classifier.predict_proba(X_test)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, probas[:, 1])
    plt.plot(recall, precision, marker='.', label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def plot_roc_curve(classifier, X_test, y_test):
    probas = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % (roc_auc))

    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='Random guessing')

    plt.plot([0, 0, 1],
             [0, 1, 1],
             linestyle=':',
             color='black',
             label='Perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.title('ROC curve')

    plt.tight_layout()
    # plt.savefig('images/06_10.png', dpi=300)
    plt.show()