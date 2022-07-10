import numpy as np
from math import sqrt
from sklearn.utils import shuffle
# from Online_Linear_Regression_Cell import concat_yt1_and_yt

def concat_yt1_and_yt(data_yt_1, data_yt):
    """
    Concatenate the y_t_1 and y_t together with shape (num_series, num_steps, 2, num_inputs)
    :param data_yt_1: y_t_1
    :param data_yt: y_t
    """
    cat = []
    for ii, (yt1, yt) in enumerate(zip(data_yt_1, data_yt)):
        num_steps = np.shape(data_yt_1[ii])[0]
        num_inputs = np.shape(data_yt_1[ii])[1]
        cat_yt1_yt = np.zeros((num_steps, 2, num_inputs))
        cat_yt1_yt[:, 0, :] = yt1
        cat_yt1_yt[:, 1, :] = yt
        cat.append(cat_yt1_yt)
    return cat

def delta_brownian(K, dt, N):
    """
    Generate Brownian motion with size (K, N),

    The Brownian motion follows: W_{t+\detla t} - W_{t} \sim \mathcal{N} (0, \delta t)
    -----------------------------------------------------------------
    K: the number of brownian motions you want to generate
    dt: float, the time interval, unit could be anything, like minute, hour, day, month.
    N: number of time steps you want to generate for each brownian motion, which mean the size of random noises.

    Returns: numpy array of size (K, N).
    """
    return np.random.normal(size=(K,) + (N,), loc=0.0, scale=sqrt(dt))

def brownian_motion_matrix(dt, RandomState, size_=[None]):
    """
    Generate brownian motion for Decoder layer 2, layer 1, layer 0.
    :param dt: delta t
    :param RandomState: RandomState instance, generate random normal values with fixed seed
    :param number_steps: expand_steps of Decoder
    :param batch_size: batch size
    :param input_size_1: input_size of the first layer in Encoder
    :param input_size_2: input_size of the second layer in Encoder
    :param input_size_3: input_size of the third layer in Encoder
    :return: dW_0, dW_1, dW_2, the brownian noises for Decoder 3 layers
    """
    dW_ = RandomState.normal(size=size_, loc=0.0, scale=sqrt(dt))
    # dW_1 = RandomState.normal(size=(expand_steps, num_noise, batch_size, input_size_2, 1), loc=0.0, scale=sqrt(dt))
    # dW_2 = RandomState.normal(size=(expand_steps, num_noise, batch_size, input_size_3, 1), loc=0.0, scale=sqrt(dt))
    return dW_

#---------------------------------------------------------------------------------------#
#                        Generate dataset for RegCell & PredCell
#---------------------------------------------------------------------------------------#

def Generate_Dataset_for_RegPredCell(data, currency='EURCNY_Curncy',
                                     len_serie=None, interval=None, LossCalculation_steps=None, future_steps=None):
    """
    Generate dataset for RegPredCell network, the input of the Dataset is the time serie with specified size
                                              the label of the Dataset is future steps you want to predict

    :param data: DataFrame, the complete FX data from csv file, e.g. 18 years' time series for different currency.
    :param currency: string, the name of the currency, should be one of ['EURCNY_Curncy', 'EURUSD_Curncy', 'EURGBP_Curncy']
    :param len_serie: intm the length of generated input serie in dataset
    :param interval: int, the size of moving window when you generate the dataset,
                     e.g. you clip 5 years' fx rates from the beginning of your long time serie,
                     then you move 30 steps (dt) forward, use the new start to generate another
                     5 years' data, which will be seen as a new input in dataset.
    :param LossCalculation_steps: int, the known steps used to calculate the loss for bayesian optimization.
    :param future_steps: int, the future steps used as labels,
                         notice that we should assume that these future steps are unknown.

    :return:
    """

    try:
        data_currency = data[currency]
    except NameError:
        print("The currency name {}".format(currency) + "is currently not in the csv file" + "\n"
              + "please choose one of the following: 'EURCNY_Curncy', 'EURUSD_Curncy', 'EURGBP_Curncy'. 谢谢!")

    # counter
    ii = 0

    x_list = []
    LossCalculation_list_y = []
    future_y_list = []

    while len_serie + (interval*ii) + LossCalculation_steps + future_steps <= len(data):

        x_t = np.array(np.copy(data[currency]
                               [(interval*ii): (interval*ii)+len_serie])).reshape((-1, 1))

        # x_t used to calculate the loss
        LossCalculation_y = np.array(np.copy(data[currency]
                                    [(interval*ii)+len_serie : (interval*ii)+len_serie + LossCalculation_steps])).reshape((-1, 1))

        # future x_t used as label
        future_y_t = np.array(np.copy(data[currency]
                                      [(interval*ii)+len_serie+LossCalculation_steps :
                                       (interval*ii)+len_serie+LossCalculation_steps + future_steps])).reshape((-1, 1))

        x_list.append(x_t)
        LossCalculation_list_y.append(LossCalculation_y)
        future_y_list.append(future_y_t)

        ii += 1

    return x_list, LossCalculation_list_y, future_y_list

def Generate_Dataset_for_AvgLossRegPredCell(data, currency='EURCNY_Curncy',
                                     len_serie=None, interval=None, LossCalculation_steps=None, future_steps=None):
    """
    Generate dataset for RegPredCell network trained by average loss, the input of the Dataset is the time serie with specified size
                                              the label of the Dataset is future steps you want to predict

    :param data: DataFrame, the complete FX data from csv file, e.g. 18 years' time series for different currency.
    :param currency: string, the name of the currency, should be one of ['EURCNY_Curncy', 'EURUSD_Curncy', 'EURGBP_Curncy']
    :param len_serie: intm the length of generated input serie in dataset
    :param interval: int, the size of moving window when you generate the dataset,
                     e.g. you clip 5 years' fx rates from the beginning of your long time serie,
                     then you move 30 steps (dt) forward, use the new start to generate another
                     5 years' data, which will be seen as a new input in dataset.
    :param LossCalculation_steps: int, the known steps used to calculate the loss for bayesian optimization.
    :param future_steps: int, the future steps used as labels,
                         notice that we should assume that these future steps are unknown.

    :return:
    """

    try:
        data_currency = data[currency]
    except NameError:
        print("The currency name {}".format(currency) + "is currently not in the csv file" + "\n"
              + "please choose one of the following: 'EURCNY_Curncy', 'EURUSD_Curncy', 'EURGBP_Curncy'. 谢谢!")

    # counter
    ii = 0

    TRAIN_X = []
    TRAIN_Y_LossCalculation = []
    TEST_Y = []

    while len_serie + (interval*ii) + LossCalculation_steps + future_steps <= len(data):
        # x_t used to calculate the loss
        train_x_t = np.array(np.copy(data[currency]
                               [(interval * ii): (interval * ii) + len_serie])).reshape((-1, 1))

        train_y_LossCalculation = \
            np.array(np.copy(data[currency][(interval * ii) : (interval * ii) +
                                                              len_serie + LossCalculation_steps])).reshape((-1, 1))

        # future x_t used as label
        test_y = np.array(np.copy(data[currency]
                                      [(interval*ii)+len_serie+LossCalculation_steps :
                                       (interval*ii)+len_serie+LossCalculation_steps + future_steps])).reshape((-1, 1))

        TRAIN_X.append(train_x_t)
        TRAIN_Y_LossCalculation.append(train_y_LossCalculation)
        TEST_Y.append(test_y)

        ii += 1

    return np.log(TRAIN_X), np.array(TRAIN_Y_LossCalculation), np.array(TEST_Y)

def preprocess_training_data(serie_steps, LossCalculation_steps, data_, data_LossCalculation, labels_):
    """
    Preprocess data for convenient training

    :param data_: the dataset
    :param labels_: the labels of dataset
    --------------------------------------
    you dont need the list_ anymore,
    because there is no training, validation, test sets difference anymore
    --------------------------------------

    :return: x_: the input vector for Online regression when you train using Bayesian optimization
             y_LossCalculation: the label for calculating the loss (f(x)) for Bayesian optimization
             x_ForPred: the input vector for online regression when you want to predict the future steps
             y_: the future steps
    """
    # For Bayesian optimization training
    # log to ensure positive
    x_ = np.log(np.array(data_))
    y_LossCalculation = np.array(data_LossCalculation)

    # For future step prediction
    # log to ensure positive
    x_ForPred = np.concatenate((x_, np.log(y_LossCalculation)), axis=1)
    y_LossCalculation = np.transpose(y_LossCalculation, [1, 0, 2])
    y_ = np.transpose(np.array(labels_), [1, 0, 2])

    x_t_1 = x_[:, :(serie_steps - 1), :]  # data in range [:num_steps-1]
    x_t = x_[:, 1:serie_steps, :]  # data in range [1:num_steps]
    x_ = concat_yt1_and_yt(x_t_1, x_t)

    x_ForPred_t_1 = x_ForPred[:, :(serie_steps+LossCalculation_steps - 1), :]  # data in range [:num_steps-1]
    x_ForPred_t = x_ForPred[:, 1:serie_steps+LossCalculation_steps, :]  # data in range [1:num_steps]
    x_ForPred = concat_yt1_and_yt(x_ForPred_t_1, x_ForPred_t)

    return x_, y_LossCalculation, x_ForPred, y_

def preprocess_training_data_Avg(serie_steps, LossCalculation_steps, data_, data_LossCalculation, labels_):
    """
    Preprocess data for convenient training

    :param data_: the dataset
    :param labels_: the labels of dataset
    --------------------------------------
    you dont need the list_ anymore,
    because there is no training, validation, test sets difference anymore
    --------------------------------------

    :return: x_: the input vector for Online regression when you train using Bayesian optimization
             y_LossCalculation: the label for calculating the loss (f(x)) for Bayesian optimization
             x_ForPred: the input vector for online regression when you want to predict the future steps
             y_: the future steps
    """
    # For Bayesian optimization training
    # log to ensure positive
    x_ = np.log(data_)
    y_LossCalculation = data_LossCalculation

    # For future step prediction
    # log to ensure positive
    x_ForPred = np.concatenate((x_, np.log(y_LossCalculation[:, -1])), axis=1)
    y_LossCalculation = np.transpose(y_LossCalculation, [2, 0, 1, 3])
    y_ = np.transpose(np.array(labels_), [1, 0, 2])

    x_t_1 = x_[:, :(serie_steps - 1), :]  # data in range [:num_steps-1]
    x_t = x_[:, 1:serie_steps, :]  # data in range [1:num_steps]
    x_ = concat_yt1_and_yt(x_t_1, x_t)

    x_ForPred_t_1 = x_ForPred[:, :(serie_steps+LossCalculation_steps - 1), :]  # data in range [:num_steps-1]
    x_ForPred_t = x_ForPred[:, 1:serie_steps+LossCalculation_steps, :]  # data in range [1:num_steps]
    x_ForPred = concat_yt1_and_yt(x_ForPred_t_1, x_ForPred_t)

    return x_, y_LossCalculation, x_ForPred, y_

def single_sample_preprocess_training_data(serie_steps, LossCalculation_steps, data_, data_LossCalculation, labels_):
    """
    Preprocess data for convenient training

    :param data_: the dataset
    :param labels_: the labels of dataset
    --------------------------------------
    you dont need the list_ anymore,
    because there is no training, validation, test sets difference anymore
    --------------------------------------

    :return: x_: the input vector for Online regression when you train using Bayesian optimization
             y_LossCalculation: the label for calculating the loss (f(x)) for Bayesian optimization
             x_ForPred: the input vector for online regression when you want to predict the future steps
             y_: the future steps
    """
    # For Bayesian optimization training
    # log to ensure positive
    x_ = [np.log(each_) for each_ in data_]
    # x_ = np.log(np.array(data_))
    y_LossCalculation = np.array(data_LossCalculation)

    # For future step prediction
    # log to ensure positive
    # x_ForPred = np.concatenate((x_, np.log(y_LossCalculation)), axis=1)
    x_ForPred = np.concatenate((x_[-1], np.log(y_LossCalculation[-1])), axis=0)
    y_LossCalculation = np.expand_dims(np.transpose(y_LossCalculation, [1, 0, 2]), axis=0)
    y_ = np.expand_dims(np.transpose(np.expand_dims(labels_, axis=0), [1, 0, 2]), axis=0)

    x_t_1 = [each_[:len(each_)-1, :] for each_ in x_]
    # x_t_1 = x_[:, :(serie_steps - 1), :]  # data in range [:num_steps-1]
    x_t = [each_[1:len(each_), :] for each_ in x_]
    # x_t = x_[:, 1:serie_steps, :]  # data in range [1:num_steps]
    x_ = concat_yt1_and_yt(x_t_1, x_t)

    x_ForPred_t_1 = np.expand_dims(x_ForPred[:(serie_steps+LossCalculation_steps - 1), :], axis=0)  # data in range [:num_steps-1]
    x_ForPred_t = np.expand_dims(x_ForPred[1:serie_steps+LossCalculation_steps, :], axis=0)  # data in range [1:num_steps]
    x_ForPred = np.array(concat_yt1_and_yt(x_ForPred_t_1, x_ForPred_t))

    return x_, y_LossCalculation, x_ForPred, y_


class BatchGenerator(object):

    def __init__(self, batch_size, shuffle_=False):

        self.batch_size = batch_size
        self.shuffle_ = shuffle_
        self.minibatch_pos = 0 # start position of a new batch

    def get_batches(self, series, LossCalculation_y, x_ForPred, labels_y):
        """

        :param series: the input vector for Online regression when you train using Bayesian optimization
        :LossCalculation_y: the label for calculating the loss (f(x)) for Bayesian optimization
        :param x_ForPred: the input vector for online regression when you want to predict the future steps
        :param labels_y: the future steps
        :param expand_steps: the steps you want to predict, just for the case that sometimes you want to predict less steps.
        """
        self.len_data = len(series)
        self.num_mini_batch = self.len_data // self.batch_size

        # random shuffle serie
        if self.shuffle_:
            series, LossCalculation_y, labels_y = shuffle(series, LossCalculation_y, labels_y)

        # Transpose just for convenient loss calculation,
        # because the predicted y from tf.scan has shape [time_steps, batch, input_size]
        # LossCalculation_y = np.transpose(LossCalculation_y, [1, 0, 2])
        # labels_y = np.transpose(labels_y, [1, 0, 2])

        for num_b in range(self.num_mini_batch):

            batch_x = np.array(series[self.minibatch_pos : self.minibatch_pos + self.batch_size])
            batch_LossCalculation_y = np.array(LossCalculation_y[:, self.minibatch_pos : self.minibatch_pos + self.batch_size])
            batch_x_ForPred = np.array(x_ForPred[self.minibatch_pos : self.minibatch_pos + self.batch_size])
            batch_y = np.array(labels_y[:, self.minibatch_pos : self.minibatch_pos + self.batch_size])

            # moving the position with batch size
            self.minibatch_pos += self.batch_size

            yield batch_x, batch_LossCalculation_y, batch_x_ForPred, batch_y

def generate_bounds_ForBayesianOpt(layer_idx, size, A_bound=None, N_bound=None, Sigma_bound=None,
                                   lr_bound=None, phi_rho_bound=None, if_ANSigma=True):
    """
    Build the bounds for input vector x of Bayesian optimization
    the Bayesian optimization algorithm will only try to find optimum
    of each element in x within the bounds.
    :param layer_idx: the number of Encoder layer, used for naming the element in x
    :param size: the size of Z, or size of A, N, Sigma
    :param ANSigma_bound: the upper bound or negative lower bound of A, N, Sigma
    :param lr_bound: the upper bound of eta_A, eta_N, eta_Sigma
    :param phi_rho_bound: the upper bound of eta_phi, eta_rho
    :return: a dict, keys are the name of elements in x, values are the lower and upper bounds of them
    """
    # build Eta vector for 5 learning rates
    Eta_ = dict({'eta_A{}_t'.format(layer_idx): (lr_bound[0], lr_bound[1]),
                 'eta_N{}_t'.format(layer_idx): (lr_bound[0], lr_bound[1]),
                 'eta_Sigma_{}t'.format(layer_idx): (lr_bound[0], lr_bound[1]),
                 'phi_{}t'.format(layer_idx): (phi_rho_bound[0], phi_rho_bound[1]),
                 'rho_{}t'.format(layer_idx): (phi_rho_bound[0], phi_rho_bound[1])})

    A_ = {}
    N_ = {}
    Sigma_ = {}

    if layer_idx != 1 and size > 1 and if_ANSigma: # if_ANSigma means if you want to generate A,N,Sigma
        # build a matrix
        for i_ in range(1, size+1):
            N_.update({'N{}_0_{}'.format(layer_idx, i_): (N_bound[0], N_bound[1])})
            for j_ in range(1, size+1):
                A_.update({'A{}_0_{}_{}'.format(layer_idx, i_, j_): (A_bound[0], A_bound[1])})
                Sigma_.update({'Sigma_{}0_{}_{}'.format(layer_idx, i_, j_): (Sigma_bound[0], Sigma_bound[1])})
        return {**A_, **N_, **Sigma_, **Eta_}

    elif layer_idx == 1 and size == 1:
        # build a vector
        A_.update({'A{}_0'.format(layer_idx): (A_bound[0], A_bound[1])})
        N_.update({'N{}_0'.format(layer_idx): (N_bound[0], N_bound[1])})

        # A_.update({'A{}_0'.format(layer_idx): (0.0, AN_bound)})
        # N_.update({'N{}_0'.format(layer_idx): (-AN_bound, 0.0)})
        Sigma_.update({'Sigma_{}0'.format(layer_idx): (Sigma_bound[0], Sigma_bound[1])})
        return {**A_, **N_, **Sigma_, **Eta_}

    return {**Eta_}


def rebuild_chosen_initial_state(x, batch_size, input_size_1, input_size_2, input_size_3):
    """
    When you use Bayesian optimization to find the best input vector x,
    you should rebuild the chose values in x as the initial states
    for validation or test process, some states like A2_0 (second layer A) is a matrix,
    so this function would clear up dict x and output the initial states you want.

    :param x: dict, contains the initial parameters of the network,
                    which are:
                              1st encode layer: the A, N, Sigma, 5 learning rates (3+5=8 elements)
                              2nd encode layer: the A, N, Sigma, 5 learning rates (9+3+9+5=26 elements)
                              3rd encode layer: 5 learning rates (5 elements)
                              in total 39 elements currenctly
    :param batch_size: int, could be train batch_size or validation batch_size or test batch_size.
    :return: the resized initial states.
    """
    #------------------------------------------------------------------#
    #                  Initial states for layer 1                      #
    #------------------------------------------------------------------#
    A1_0 = np.tile(x['A1_0'], (batch_size, input_size_1, input_size_1))
    N1_0 = np.tile(x['N1_0'], (batch_size, input_size_1, 1))
    Sigma_10 = np.tile(x['Sigma_10'], (batch_size, input_size_1, input_size_1))

    eta_A1_t = np.tile(x['eta_A1_t'], (batch_size, input_size_1, input_size_1))
    eta_N1_t = np.tile(x['eta_N1_t'], (batch_size, input_size_1, 1))
    eta_Sigma_1t = np.tile(x['eta_Sigma_1t'], (batch_size, input_size_1, input_size_1))
    phi_1t = np.tile(x['phi_1t'], (batch_size, input_size_1, 1))
    rho_1t = np.tile(x['rho_1t'], (batch_size, input_size_1, input_size_1))

    #------------------------------------------------------------------#
    #                   Initial states for layer 2                     #
    #------------------------------------------------------------------#
    A2_0 = np.tile(np.array([[x['A2_0_11'], x['A2_0_12'], x['A2_0_13']],
                             [x['A2_0_21'], x['A2_0_22'], x['A2_0_23']],
                             [x['A2_0_31'], x['A2_0_32'], x['A2_0_33']]]), (batch_size, 1, 1))

    N2_0 = np.tile(np.array([ [x['N2_0_1']], [x['N2_0_2']], [x['N2_0_3']] ]), (batch_size, 1, 1))
    #
    Sigma_20 = np.tile(np.array([[x['Sigma_20_11'], x['Sigma_20_12'], x['Sigma_20_13']],
                                 [x['Sigma_20_21'], x['Sigma_20_22'], x['Sigma_20_23']],
                                 [x['Sigma_20_31'], x['Sigma_20_32'], x['Sigma_20_33']]]), (batch_size, 1, 1))

    eta_A2_t = np.tile(x['eta_A2_t'], (batch_size, input_size_2, input_size_2))
    eta_N2_t = np.tile(x['eta_N2_t'], (batch_size, input_size_2, 1))
    eta_Sigma_2t = np.tile(x['eta_Sigma_2t'], (batch_size, input_size_2, input_size_2))
    phi_2t = np.tile(x['phi_2t'], (batch_size, input_size_2, 1))
    rho_2t = np.tile(x['rho_2t'], (batch_size, input_size_2, input_size_2))

    #------------------------------------------------------------------#
    #                  Initial states for layer 3                      #
    #------------------------------------------------------------------#
    eta_A3_t = np.tile(x['eta_A3_t'], (batch_size, input_size_3, input_size_3))
    eta_N3_t = np.tile(x['eta_N3_t'], (batch_size, input_size_3, 1))
    eta_Sigma_3t = np.tile(x['eta_Sigma_3t'], (batch_size, input_size_3, input_size_3))
    phi_3t = np.tile(x['phi_3t'], (batch_size, input_size_3, 1))
    rho_3t = np.tile(x['rho_3t'], (batch_size, input_size_3, input_size_3))

    return A1_0, N1_0, Sigma_10, \
           eta_A1_t, eta_N1_t, eta_Sigma_1t, phi_1t, rho_1t, \
           A2_0, N2_0, Sigma_20, \
           eta_A2_t, eta_N2_t, eta_Sigma_2t, phi_2t, rho_2t, \
           eta_A3_t, eta_N3_t, eta_Sigma_3t, phi_3t, rho_3t

def rebuild_chosen_initial_state_layerwise(x, batch_size, input_size, layer=1, if_ANSigma=True):

    eta_A_t = np.tile(x['eta_A{}_t'.format(layer)], (batch_size, input_size, input_size)).astype(np.float32)
    eta_N_t = np.tile(x['eta_N{}_t'.format(layer)], (batch_size, input_size, 1)).astype(np.float32)
    eta_Sigma_t = np.tile(x['eta_Sigma_{}t'.format(layer)], (batch_size, input_size, input_size)).astype(np.float32)
    phi_t = np.tile(x['phi_{}t'.format(layer)], (batch_size, input_size, 1)).astype(np.float32)
    rho_t = np.tile(x['rho_{}t'.format(layer)], (batch_size, input_size, input_size)).astype(np.float32)

    if input_size == 1 and layer == 1:
        A_ = np.tile(x['A{}_0'.format(layer)], (batch_size, input_size, input_size)).astype(np.float32)
        N_ = np.tile(x['N{}_0'.format(layer)], (batch_size, input_size, 1)).astype(np.float32)
        Sigma_ = np.tile(x['Sigma_{}0'.format(layer)], (batch_size, input_size, input_size)).astype(np.float32)
        return A_, N_, Sigma_, eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t

    elif input_size != 1 and layer != 1:

        # if include A,N,Sigma
        if if_ANSigma:

            # first dimension is for batch_size
            A_ = np.ndarray((1, input_size, input_size), dtype=np.float32).astype(np.float32)
            N_ = np.ndarray((1, input_size, 1), dtype=np.float32).astype(np.float32)
            Sigma_ = np.ndarray((1, input_size, input_size), dtype=np.float32).astype(np.float32)

            for i_ in range(1, input_size+1):
                N_[0, i_-1] = x['N{}_0_{}'.format(layer, i_)]
                for j_ in range(1, input_size+1):
                    A_[0, i_-1, j_-1] = x['A{}_0_{}_{}'.format(layer, i_, j_)]
                    Sigma_[0, i_-1, j_-1] = x['Sigma_{}0_{}_{}'.format(layer, i_, j_)]

            A_ = np.tile(A_, (batch_size, 1, 1)).astype(np.float32)
            N_ = np.tile(N_, (batch_size, 1, 1)).astype(np.float32)
            Sigma_ = np.tile(Sigma_, (batch_size, 1, 1)).astype(np.float32)
            return A_, N_, Sigma_, eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t

        else:
            return eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t


def confident_interval_accuracy(series, upper_bound, lower_bound):
    """
    Calculate the accuracy of confidential interval by evaluating how many points are inside of the interval
    :param series: the label time series
    :param upper_bound: predicted upper bound
    :param lower_bound: predicted lower bound
    :return: percentage accuracy
    """
    how_many = 0
    for ii in range(len(series)):
        if series[ii] >= lower_bound[ii] and series[ii] <= upper_bound[ii]:
            how_many += 1
    return how_many/len(series) * 100

