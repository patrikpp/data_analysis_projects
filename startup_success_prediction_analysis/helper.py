import numpy as np
from sklearn import metrics


def evaluate_classifier(y_test, y_pred):
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}')
    print(f'Recall: {metrics.recall_score(y_test, y_pred):.4f}')
    print(f'Precision: {metrics.precision_score(y_test, y_pred):.4f}')
    print(f'F1: {metrics.f1_score(y_test, y_pred):.4f}')


def print_feature_importances(tree_classifier, X_columns):
    importances = tree_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(len(X_columns)):
        print("%2d) %-*s %f" % (f + 1, 30, X_columns[indices[f]], importances[indices[f]]))
