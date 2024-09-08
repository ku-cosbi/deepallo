#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np


def evaluation(y_labels, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_labels, y_preds).ravel()

    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    specifity = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    roc_auc = roc_auc_score(y_labels, y_preds)

    return precision, accuracy, recall, specifity, f1, roc_auc


def model_eval(x_val, y_val, model):
    test_true = []
    test_pred = []

    for x, y_true in zip(x_val, y_val):
        x = np.array(x).reshape(-1, 19)
        y_score = model.predict(x)
        # take max score as positive
        y_pred = np.zeros_like(y_score).astype(int)
        y_pred[y_score.argmax(0)] = 1
        # add to results
        test_true += y_true
        test_pred += y_pred.tolist()

    return evaluation(test_true, test_pred)
