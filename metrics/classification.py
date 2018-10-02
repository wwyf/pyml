import numpy as np


def precision_score(y_true, y_pred):
    """Compute the precision

    适用于分类问题，直接计算预测出来的Y的正确率

    Parameters
    --------------

    y_true : 1d array-like
        Ground truth (correct) target values.

    y_pred : 1d array-like
        Estimated targets as returned by a classifier.

    Return
    -------
        accuracy_rate : double
    """
    assert(len(y_pred) == len(y_true))
    assert(len(y_pred.shape) == 1)
    assert(len(y_true.shape) == 1)
    total_num = len(y_pred)
    success_num = 0
    for i in range(0, total_num):
        if (y_true[i] == y_pred[i]):
            success_num += 1
    return float(success_num)/total_num

