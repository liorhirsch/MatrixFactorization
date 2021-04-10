import numpy as np


def rmse(vals):
    vals = np.array(vals)
    return np.sqrt(np.mean(vals ** 2))

def mae(vals):
    vals = np.array(vals)
    return np.mean(np.abs(vals))