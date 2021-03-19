import numpy as np


def rmse(true_val, preds):
    return np.sum((true_val - preds) ** 2)