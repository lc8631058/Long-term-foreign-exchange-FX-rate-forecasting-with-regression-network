import numpy as np
from scipy import stats
from dtw import dtw

def MDA(y_pred, y_true):
    """Mean direcetional accuracy"""
    return np.mean((np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])).astype(int))

def RMSE(y_pred, y_true):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def MASE(training_series, prediction_series, testing_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    """

    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d

def MAPE(y_pred, y_true):
    """
    mean absolute percentage error
    :param y_pred: 1 single predicted series
    :param y_true: 1 single label series
    :return: MAPE
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))

def R(y_pred, y_true):
    """
    pearson R
    :param y_pred: 1 single predicted series
    :param y_true: 1 single label series
    :return: R score
    """
    # mean_y_true = np.mean(y_true)
    # mean_y_pred = np.mean(y_pred)
    r, prob = stats.pearsonr(y_pred, y_true)
    return r
    # return np.sum(np.multiply((y_true - mean_y_true), (y_pred - mean_y_pred))) / \
           # np.multiply(np.sqrt(np.sum(np.square(y_true - mean_y_true))),
           #             np.sqrt(np.sum(np.square(y_pred - mean_y_pred)))
           #             )

def SpearmanR(y_pred, y_true):
    """
    spearman R
    :param y_pred:
    :param y_true:
    :return:
    """
    rho, prob = stats.spearmanr(y_pred, y_true)
    return rho

def KendallTau(y_pred, y_true):
    """
    kindall's Tau
    :param y_pred:
    :param y_true:
    :return:
    """
    tau, prob = stats.kendalltau(y_pred, y_true)
    return tau

def DTW(y_pred, y_true):
    x = np.array(y_pred).reshape(-1, 1)
    y = np.array(y_true).reshape(-1, 1)

    manhattan_distance = lambda x, y: np.abs(x - y)

    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)

    return d

def TheilU(y_pred, y_true):
    """
    Theil U indicator
    :param y_pred: predictions
    :param y_true: labels
    :return: Theil U
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred))) / \
           (np.sqrt(np.mean(np.square(y_true))) + np.sqrt(np.mean(np.square(y_pred))))

def MAE(y_pred, y_true):
    """
    Mean absolute error
    :param y_pred:
    :param y_true:
    :return:
    """
    return np.mean(np.abs(y_true - y_pred))

def AbsDev(y_pred, y_true):
    """
    AbsDev indicator
    :param y_pred:
    :param y_true:
    :return:
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
