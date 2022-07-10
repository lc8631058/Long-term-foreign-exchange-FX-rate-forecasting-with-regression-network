import random
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

get_rid_abnormal = False
# get_rid_abnormal = True
if get_rid_abnormal:

    with open("./Data/artificial_fx_2/for_training_OnlineLR/num_abnorms.pkl", 'rb') as f:
        abnorm_files = pkl.load(f)

    good_series = []
    good_labels = []
    for f_ in range(100):

        with open("./Data/artificial_fx_2/for_training_OnlineLR/arti_{}".format(f_), 'rb') as f:
            da_ = pkl.load(f)
        with open("./Data/artificial_fx_2/for_training_OnlineLR/paras_{}".format(f_), 'rb') as f:
            pa_ = pkl.load(f)

        good_series.extend(np.delete(da_, abnorm_files[f_], axis=0))
        good_labels.extend(np.delete(pa_, abnorm_files[f_], axis=0))

    with open("./Data/artificial_fx_2/for_training_OnlineLR/good_series.pkl", 'wb') as f:
        pkl.dump(good_series, f)
    with open("./Data/artificial_fx_2/for_training_OnlineLR/good_labels.pkl", 'wb') as f:
        pkl.dump(good_labels, f)


draw_abnormal = False
# draw_abnormal = True

ab_file_num = 29
ab_serie_num = 91

if draw_abnormal:
    with open("./Data/artificial_fx_2/for_training_OnlineLR/arti_{}".format(ab_file_num), 'rb') as f:
        abnorm_file = pkl.load(f)
    with open("./Data/artificial_fx_2/for_training_OnlineLR/paras_{}".format(ab_file_num), 'rb') as f:
        abnorm_labels = pkl.load(f)
    abnorm_serie = abnorm_file[ab_serie_num]
    abnorm_label = abnorm_labels[ab_serie_num]

    norm_serie = abnorm_file[ab_serie_num+1]
    norm_label = abnorm_labels[ab_serie_num+1]

    t_ = np.linspace(0, 1829, num=1830)
    plt.figure()
    plt.title('Abnormal serie')
    plt.plot(t_, abnorm_serie[:, 0], label='USD')
    plt.plot(t_, abnorm_serie[:, 1], label='GBP')
    plt.plot(t_, abnorm_serie[:, 2], label='CNY')
    plt.figure()
    plt.title('normal serie')
    plt.plot(t_, norm_serie[:, 0], label='USD')
    plt.plot(t_, norm_serie[:, 1], label='GBP')
    plt.plot(t_, norm_serie[:, 2], label='CNY')
    plt.grid()
    plt.legend()
    plt.show()

class OnlineLR_cell(object):

    def __init__(self, input_size, state_size, trainable=True, eta_max=0.2, eta_min=0.0, seed=1):

        self.input_size = input_size
        self.state_size = state_size # hidden state size, means h_t = [eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t]
        self.trainable = trainable
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.seed = seed
        # --------- weights ---------

        # forget gate

        self.W_f = tf.Variable( tf.truncated_normal([2*self.input_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.U_f = tf.Variable( tf.truncated_normal([self.state_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.b_f = tf.Variable( tf.zeros([self.state_size]),
                                trainable=self.trainable )

        # input gate
        self.W_i = tf.Variable(tf.truncated_normal([2*self.input_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.U_i = tf.Variable(tf.truncated_normal([self.state_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.b_i = tf.Variable(tf.zeros([self.state_size]),
                                trainable=self.trainable )

        # output gate
        self.W_o = tf.Variable(tf.truncated_normal([2*self.input_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.U_o = tf.Variable(tf.truncated_normal([self.state_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.b_o = tf.Variable(tf.zeros([self.state_size]),
                                trainable=self.trainable )

        # temporary cell state
        self.W_c = tf.Variable(tf.truncated_normal([2*self.input_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.U_c = tf.Variable(tf.truncated_normal([self.state_size, self.state_size], mean=0, stddev=.01, seed=self.seed),
                                trainable=self.trainable )
        self.b_c = tf.Variable(tf.zeros([self.state_size]),
                                trainable=self.trainable )

        # ---------------------------------
        # self.eta_A = tf.Variable( tf.random_uniform((1,), minval=0., maxval=0.3, seed=self.seed),
        #                             trainable=self.trainable, name='eta_A' )
        # self.eta_N = tf.Variable( tf.random_uniform((1,), minval=0., maxval=0.3, seed=self.seed),
        #                             trainable=self.trainable, name='eta_N' )
        # self.eta_Sigma = tf.Variable( tf.random_uniform((1,), minval=0., maxval=0.3, seed=self.seed),
        #                                 trainable=self.trainable, name='eta_Sigma' )
        # # phi for EMA_epsilon
        # self.phi = tf.Variable( tf.random_uniform((1,), minval=0., maxval=1., seed=self.seed),
        #                         trainable=self.trainable, name='phi' )
        # # rho for EMA_cov(epsilon_i, epsilon_j)
        # self.rho = tf.Variable( tf.random_uniform((1,), minval=0., maxval=1., seed=self.seed),
        #                         trainable=self.trainable, name='rho' )

        # inputs [batch, seq, input_size]
        self._inputs = tf.placeholder(tf.float32, [None, None, 2, self.input_size], name='inputs_')  # 2: y_t and y_t-1

        # initial state for tf.scan initializer
        self.A_0 = tf.placeholder(tf.float32, [None, self.input_size, self.input_size], name='A_0')
        self.N_0 = tf.placeholder(tf.float32, [None, self.input_size, 1], name='N_0')
        self.Sigma_0 = tf.placeholder(tf.float32, [None, self.input_size, self.input_size], name='Sigma_0')
        self.E_epsilon_0 = tf.placeholder(tf.float32, [None, self.input_size, 1], name='E_epsilon_0')
        self.cov_epsilon_0 = tf.placeholder(tf.float32, [None, self.input_size, self.input_size], name='cov_epsilon_0')
        self.cell_state_0 = tf.placeholder(tf.float32, [None, self.state_size], name='cell_state_0')
        self.hidden_state_0 = tf.placeholder(tf.float32, [None, self.state_size], name='hidden_state_0')
        self.y_t_pred_0 = tf.placeholder(tf.float32, [None, self.input_size, 1], name='y_t_pred')
        self.y_t_0 = tf.placeholder(tf.float32, [None, self.input_size, 1], name='y_t_true')

        # self.update_epsilon = tf.placeholder(tf.float32, [None, self.input_size, 1], name='update_epsilon')
        # self.matmul_A_Y = tf.placeholder(tf.float32, [None, self.input_size, 1], name='matmul_A_Y')
        # self.delta_t = tf.placeholder(tf.float32, [None, self.input_size, 1], name='delta_t')
        # self.y_t_label_0 = tf.placeholder(tf.float32, [None, self.input_size, 1], name='y_t_label')
        self.epsilon_t_0 = tf.placeholder(tf.float32, [None, self.input_size, 1], name='epsilon_t')

        # reshape the inputs to work with tf.scan function
        # [batch, seq, input_units] --> [seq, batch, input_units]
        self.scan_inputs = tf.transpose(self._inputs, perm=[1, 0, 2, 3], name='scan_inputs')
        self.initial_state = [self.A_0, self.N_0, self.Sigma_0, self.E_epsilon_0,
                              self.cov_epsilon_0, self.y_t_pred_0, self.y_t_0,
                              self.epsilon_t_0, self.cell_state_0, self.hidden_state_0]
                              # self.matmul_A_Y, self.delta_t, ]

    def EMA_epsilon(self, previous_EMA_epsilon, epsilon_t, phi):
        """
        calculate the EMA epsilon
        :param previous_EMA_epsilon: the EMA epsilon vector at time t-1
        :param epsilon_t: the epsilon vector at time t
        :return: the new evaluated EMA epsilon vector at time t
        """
        return tf.multiply(phi, epsilon_t) + tf.multiply((1. - phi), previous_EMA_epsilon)

    def EMA_cov_epsilon(self, previous_EMA_cov_epsilon, epsilon_t, E_epsilon_t, rho):
        """
        calculate the EMA cov(epsilon)
        :param previous_EMA_cov_epsilon: the EMA cov(epsilon) matrix at time t-1
        :param epsilon_t: the epsilon vector at time t
        :param EMA_epsilon_t: the EMA epsilon vector at time t
        :return: the new evaluated EMA cov(epsilon) matrix at time t
        """
        # transpose
        epsilon_t_T = tf.transpose(epsilon_t, perm=[0, 2, 1])
        E_epsilon_t_T = tf.transpose(E_epsilon_t, perm=[0, 2, 1])

        EMA_cov_epsilon = tf.multiply(rho, ( tf.matmul(epsilon_t, epsilon_t_T) - tf.matmul(epsilon_t, E_epsilon_t_T)
                          - tf.matmul(E_epsilon_t, epsilon_t_T) + tf.matmul(E_epsilon_t, E_epsilon_t_T) )) \
                          + tf.multiply((1. - rho), previous_EMA_cov_epsilon)

        return EMA_cov_epsilon

    def linear_activation(self, x, eta_max, eta_min):

        return 0.5 * (eta_max + eta_min) + 0.5 * (eta_max - eta_min) * x

    def OnlineLR(self, previous_states, y):
        """
        Performs the operations of online linear regression
        Designed for tf.scan, the first position "batch" will individually be processed by tf.scan

        :param previous_states: a tuple contains (A_t-1, N_t-1, Sigma_t-1)
        :param y: inputs with shape [seq, batch, 2, input_size], where y_t is the input vector at time t; 2: y_t, y_t-1
        :return: a tuple contains (A_t, N_t, Sigma_t)
        """

        # unpack the states from last time step
        A_t_1, N_t_1, Sigma_t_1, E_epsilon_t_1, cov_epsilon_t_1, pred_y_t_, y_t_true_, epsilon_t_, c_t_1, h_t_1 = previous_states
        # matmul_A_Y, delta_y_, epsilon_t

        # unpack the input y into [batch, y_t-1] and [batch, y_t]
        y_t_1, y_t = tf.unstack(y, axis=1)

        # => [batch, concat(y_t_1, y_t)]
        concat_yt1_yt = tf.concat([y_t_1, y_t], axis=1)

        y_t_1 = tf.expand_dims(y_t_1, axis=-1)
        y_t = tf.expand_dims(y_t, axis=-1)

        delta_y_t = y_t - y_t_1 # Delta y_t

        # calculate the epsilon
        epsilon_t = delta_y_t - (tf.matmul(A_t_1, y_t_1) + N_t_1)

        y_t_1T = tf.transpose(y_t_1, perm=[0, 2, 1]) # transpose
        Sigma_t_1T = tf.transpose(Sigma_t_1, perm=[0, 2, 1])

        # weights for eta LSTM
        f_t = tf.sigmoid(tf.matmul(concat_yt1_yt, self.W_f) + tf.matmul(h_t_1, self.U_f) + self.b_f)
        i_t = tf.sigmoid(tf.matmul(concat_yt1_yt, self.W_i) + tf.matmul(h_t_1, self.U_i) + self.b_i)
        o_t = tf.sigmoid(tf.matmul(concat_yt1_yt, self.W_o) + tf.matmul(h_t_1, self.U_o) + self.b_o)
        c_hat = tf.tanh(tf.matmul(concat_yt1_yt, self.W_c)
                        + tf.matmul(h_t_1, self.U_c) + self.b_c)  # c_hat: temporary cell state
        c_t = tf.multiply(f_t, c_t_1) + tf.multiply(i_t, c_hat)

        h_t = tf.multiply(o_t, tf.tanh(c_t))
        h_t = self.linear_activation(h_t, self.eta_max, self.eta_min)

        # dense layer
        eta_A_t = tf.tile(tf.expand_dims(tf.expand_dims(h_t[:, 0], axis=-1), axis=-1), [1, self.input_size, self.input_size])
        eta_N_t = tf.tile(tf.expand_dims(tf.expand_dims(h_t[:, 1], axis=-1), axis=-1), [1, self.input_size, 1])
        eta_Sigma_t = tf.tile(tf.expand_dims(tf.expand_dims(h_t[:, 2], axis=-1), axis=-1), [1, self.input_size, self.input_size])
        phi_t = tf.tile(tf.expand_dims(tf.expand_dims(1. - h_t[:, 3], axis=-1), axis=-1), [1, self.input_size,1])
        rho_t = tf.tile(tf.expand_dims(tf.expand_dims(1. - h_t[:, 4], axis=-1), axis=-1), [1, self.input_size, self.input_size])

        A_t = A_t_1 + 2. * tf.multiply(eta_A_t, tf.matmul(epsilon_t, y_t_1T))
        # N_t
        N_t = N_t_1 + 2. * eta_N_t * epsilon_t

        E_epsilon_t = self.EMA_epsilon(E_epsilon_t_1, epsilon_t, phi_t)

        cov_epsilon_t = self.EMA_cov_epsilon(cov_epsilon_t_1, epsilon_t, E_epsilon_t, rho_t)

        Sigma_t = Sigma_t_1 - 4. * tf.multiply( eta_Sigma_t, tf.matmul(
                                                (tf.matmul(Sigma_t_1, Sigma_t_1T) - cov_epsilon_t) , Sigma_t_1 ) )


        # --------- Calculate the cost --------- #

        pred_y_t = tf.matmul(A_t_1, y_t_1) + N_t_1 + y_t_1
        y_t_true = y_t

        return [A_t, N_t, Sigma_t, E_epsilon_t, cov_epsilon_t, pred_y_t, y_t_true, epsilon_t, c_t, h_t]

    def get_states(self):
        """
        tf.scan function will recursively execute the function you give to it, collect the outputs in each time step
        :return: the collected output_states with shape [seq, 5, batch_size], where 5 represents the 5 states
                 we have in OnlineLR_Cell
        """
        states_over_time = tf.scan(self.OnlineLR, self.scan_inputs, initializer=self.initial_state, name='states')

        return states_over_time

def concat_yt1_and_yt(data_yt_1, data_yt):
    """
    Concatenate the y_t_1 and y_t together with shape (num_series, num_steps, 2, num_inputs)
    :param data_yt_1: y_t_1
    :param data_yt: y_t
    """
    num_series, num_steps, num_inputs = data_yt.shape

    new_data_yt_1 = np.zeros([num_series, num_steps, 1, num_inputs])  # 2: yt and yt_1
    new_data_yt = np.zeros([num_series, num_steps, 1, num_inputs])

    new_data_yt_1[:, :, 0, :] = data_yt_1
    new_data_yt[:, :, 0, :] = data_yt
    return np.concatenate((new_data_yt_1, new_data_yt), axis=2)


# =======================================================================================#
#                         Process data for faster training
# =======================================================================================#
#
num_steps = 730 # 366*5 days
# if_Non_Nega = {'EURUSD_Curncy': True, 'EURGBP_Curncy': True, 'EURCNY_Curncy': True}
if_Non_Nega = {'EURCNY_Curncy': True}
num_factors = len(if_Non_Nega) # number of financial factors
input_size = num_factors
state_size = 5 # [eta_A, eta_N, eta_Sigma, phi, rho]
output_size = 5
len_Sigma = num_factors**2 # length of Sigma without zeros
len_Alpha = num_factors**2
len_N = num_factors
len_params = len_Alpha + len_N + len_Sigma # number of parameters to predict

# d = True
d = False
if d:
    for f_ in range(100):
        with open("./Data/artificial_fx_2/original/arti_{}".format(f_), "rb") as f:
            data_ = pkl.load(f)
        with open("./Data/artificial_fx_2/original/paras_{}".format(f_), "rb") as f:
            params_ = pkl.load(f)

        ndarray_data = np.zeros([len(data_), num_steps, input_size])  # transform data and parameters to ndarray
        ndarray_param = np.zeros([len(data_), len_params])

        for i in range(200):
            ndarray_data[i, :, :] = data_[i].values
            ndarray_param[i, :] = np.concatenate((
                params_[i][0].flatten(),
                # np.log(np.abs(params_[i][0].flatten())),
                # np.sign(params_[i][0].flatten()),
                # np.log(np.abs(params_[i][1])),
                params_[i][1],
                # np.sign(params_[i][1]),
                params_[i][2].flatten()
                # np.log(np.abs(extract_lower_triangular(params_[i][2]))),
                # np.sign(extract_lower_triangular(params_[i][2]))
            ), axis=0)
        with open("./Data/artificial_fx_2/for_training_OnlineLR/arti_{}".format(f_), "wb") as f:
            pkl.dump(ndarray_data, f)
        with open("./Data/artificial_fx_2/for_training_OnlineLR/paras_{}".format(f_), "wb") as f:
            pkl.dump(ndarray_param, f)

#=======================================================================================#
#                                    Training
#=======================================================================================#
#
class FutureBatchGenerator(object):

    def __init__(self, batch_size, shuffle_=True):

        self.batch_size = batch_size
        self.shuffle_ = shuffle_
        self.pos = 0 # start position of moving window

    def get_batches(self, series, labels_A, labels_N, labels_Sigma):
        self.len_data = len(series)
        self.num_mini_batch = self.len_data // self.batch_size

        # random shuffle series
        if self.shuffle_:
            series, labels_A, labels_N, labels_Sigma = shuffle(series, labels_A, labels_N, labels_Sigma)

        for num_b in range(self.num_mini_batch):

            x_ = np.array(series[self.pos : self.pos + self.batch_size])
            y_A = np.array(labels_A[self.pos : self.pos + self.batch_size])
            y_N = np.array(labels_N[self.pos : self.pos + self.batch_size])
            y_Sigma = np.array(labels_Sigma[self.pos : self.pos + self.batch_size])

            self.pos += self.batch_size

            yield x_, y_A, y_N, y_Sigma


def Optimization(OnlineLR_cell, input_size, state_size, eta_max, eta_min,
                 lr_1=0.1, lr_2=0.1, lr_3=0.1, trainable=True, rd_seed=None):

    rnn = OnlineLR_cell(input_size, state_size,
                        trainable=trainable, eta_max=eta_max, eta_min=eta_min, seed=rd_seed)

    output_states = rnn.get_states()

    label_A = tf.placeholder(tf.float32, [None, input_size*input_size], name='label_A')
    label_N = tf.placeholder(tf.float32, [None, input_size], name='label_N')
    label_Sigma = tf.placeholder(tf.float32, [None, input_size*input_size], name='label_Sigma')

    # output_states[7]: pred_y_t;   output_states[8]: y_t_true
    # error_A_N_ = output_states[7] - output_states[8]
    error_A_N_ = output_states[5] - output_states[6]

    # tf.transpose: [seq, batch, input_size, 1] -> [seq, batch, 1, input_size]
    # tf.matmul: only inner two dimension take matrix multiplication
    cost_A_N_list = tf.matmul(tf.transpose(error_A_N_, perm=[0, 1, 3, 2]), error_A_N_)
    cost_A_N_ = tf.reduce_mean( cost_A_N_list )

    # loss A
    # pred_A_ = output_states[0] # [seq, batch, input_size, input_size]
    # loss_A_ = tf.square(tf.norm(tf.reshape(pred_A_, [tf.shape(pred_A_)[0], tf.shape(pred_A_)[1], rnn.input_size*rnn.input_size]) - label_A))
    # mean_loss_A_ = tf.reduce_mean(loss_A_)
    label_A_mat = tf.expand_dims( tf.reshape(label_A, [tf.shape(label_A)[0], input_size, input_size]), axis=0)
    label_A_mat = tf.tile(label_A_mat, [num_steps - 1, 1, 1, 1])
    pred_A = output_states[0]
    error_A = tf.norm( label_A_mat - pred_A )
    cost_A = tf.reduce_mean( error_A )

    compare_A = tf.concat([tf.squeeze(label_A_mat[-1],axis=-1), tf.squeeze(pred_A[-1],axis=-1)], axis=-1)

    label_N_mat = tf.expand_dims( tf.reshape(label_N, [tf.shape(label_N)[0], input_size, 1]), axis=0)
    label_N_mat = tf.tile(label_N_mat, [num_steps - 1, 1, 1, 1])
    pred_N = output_states[1]
    error_N = tf.norm( label_N_mat - pred_N )
    cost_N = tf.reduce_mean( error_N )

    compare_N = tf.concat([tf.squeeze(label_N_mat[-1],axis=-1), tf.squeeze(pred_N[-1],axis=-1)], axis=-1)

    label_Sigma_mat = tf.expand_dims( tf.reshape(label_Sigma, [tf.shape(label_Sigma)[0], input_size, input_size]), axis=0)
    # label_Sigma_mat = tf.expand_dims( rnn.label_Sigma, axis=0)


    # duplicate matrix from [batch, 3, 3] to [num_steps, batch, 3, 3]
    # e.g. inputs:[1,3], second position = [3,2] ==> outputs:[3,6]
    label_Sigma_mat = tf.tile(label_Sigma_mat, [num_steps - 1, 1, 1, 1])

    pred_Sigma = output_states[2]

    label_SST = tf.matmul(label_Sigma_mat, tf.transpose(label_Sigma_mat, [0, 1, 3, 2])) # SST: Sigma * Sigma^{T}
    pred_SST = tf.matmul(pred_Sigma, tf.transpose(pred_Sigma, [0, 1, 3, 2]))
    error_Sigma = tf.norm(label_SST - pred_SST)


    cost_Sigma = tf.reduce_mean( error_Sigma )

    # compare_Sigma = tf.concat([tf.squeeze(label_SST[-1],axis=-1), tf.squeeze(pred_SST[-1],axis=-1)], axis=-1)
    compare_Sigma = tf.concat([tf.square(tf.squeeze(label_Sigma_mat[-1], axis=-1)),
                               tf.square(tf.squeeze(pred_Sigma[-1], axis=-1))], axis=-1)

    # Optimizers
    optimizer_A = tf.train.AdamOptimizer(lr_1).minimize(cost_A)
    optimizer_N = tf.train.AdamOptimizer(lr_2).minimize(cost_N)
    optimizer_Sigma = tf.train.AdamOptimizer(lr_3).minimize(cost_Sigma)


    # Metric: RRSE
    RRSE_A = tf.sqrt(tf.reduce_mean(tf.square(pred_A[-1] - label_A_mat[-1])) / denominator_RRSE['A'])
    RRSE_N = tf.sqrt(tf.reduce_mean(tf.square(pred_N[-1] - label_N_mat[-1])) / denominator_RRSE['N'])
    #RRSE_Sigma = tf.sqrt(tf.reduce_mean(tf.square(pred_SST[-1] - label_SST[-1])) / denominator_RRSE['Sigma'])
    RRSE_Sigma = tf.sqrt(tf.reduce_mean( tf.square(tf.square(tf.squeeze(label_Sigma_mat[-1], axis=-1))
                        - tf.square(tf.squeeze(pred_Sigma[-1], axis=-1)) )) / denominator_RRSE['Sigma'])

    return rnn, output_states, label_A, label_N, label_Sigma, compare_A, compare_N, compare_Sigma, \
           cost_A, cost_N, cost_Sigma, RRSE_A, RRSE_N, RRSE_Sigma, optimizer_A, optimizer_N, optimizer_Sigma


def initialization(batch_size, input_size):
    # Initial states
    # A_0 = np.random.rand(batch_size, input_size, input_size)
    # N_0 = np.random.rand(batch_size, input_size, 1)
    A_0 = np.zeros([batch_size, input_size, input_size])
    #
    # A_0 = - np.array([np.eye(input_size)] * batch_size)
    N_0 = np.zeros([batch_size, input_size, 1])

    # Sigma_0 = np.random.rand(batch_size, input_size, input_size)
    Sigma_0 = np.array([np.eye(input_size)] * batch_size)

    E_epsilon_0 = np.zeros([batch_size, input_size, 1])
    cov_epsilon_0 = np.zeros([batch_size, input_size, input_size])

    cell_state_0 = np.zeros([batch_size, state_size])
    hidden_state_0 = np.zeros([batch_size, state_size])

    y_t_pred_0 = np.zeros([batch_size, input_size, 1])
    y_t_0 = np.zeros([batch_size, input_size, 1])
    epsilon_t_0 = np.zeros([batch_size, input_size, 1])

    return A_0, N_0, Sigma_0, E_epsilon_0, cov_epsilon_0, cell_state_0, hidden_state_0, y_t_pred_0, y_t_0, epsilon_t_0

def preprocess_training_data(num_steps, data_, labels_, list_):
    """
    Preprocess data for convenient training
    :param data_: the dataset
    :param labels_: the labels of dataset
    :param list_: a list of numbers, determines which time series to choose
    :return: x_, y_A, y_N, y_Sigma
    """
    x_ = np.log(np.array([data_[i] for i in list_]))
    y_ = np.array([labels_[i] for i in list_])

    data_t_1 = x_[:, :(num_steps - 1), :]  # data in range [:num_steps-1]
    data_t = x_[:, 1:num_steps, :]  # data in range [1:num_steps]
    x_ = concat_yt1_and_yt(data_t_1, data_t)

    y_A = y_[:, 0]
    y_N = y_[:, 1]
    y_Sigma = y_[:, 2]
    # train_y_A = np.reshape(np.expand_dims(train_y_A, axis=0), [1, input_size * input_size])
    # train_y_N = np.reshape(np.expand_dims(train_y_N, axis=0), [1, input_size])
    # train_y_Sigma = np.reshape(np.expand_dims(train_y_Sigma, axis=0), [1, input_size * input_size])
    return x_, y_A, y_N, y_Sigma

#---------------------------------------------------------------------------------------#
#                                  Start Training
#---------------------------------------------------------------------------------------#
#
lr_1 = 1e-03 * 1
lr_2 = 1e-03 * 1
lr_3 = 1e-03 * 1
epochs = 800
rd_seed = 5
batch_size = 500
val_batch_size = 50
shuffel = True
eta_max = 0.3
eta_min = 0

# Strat_ = True
Strat_ = False
if Strat_:
    random.seed(a=1)
    path = "./Data/artificial_fx_2/predict_future_dataset"
    serie_list = np.arange(400)

    train_list = random.sample(serie_list.tolist(), 300)
    val_list = random.sample([e for e in serie_list if e not in train_list], 50)
    test_list = [e for e in serie_list if e not in train_list + val_list]

    with open(path+'denominator_RRSE', 'rb') as f:
        denominator_RRSE = pkl.load(f)
    with open(path + '/single_serie_for_OnlineLR_CNY.pkl', "rb") as f:
        data_ = pkl.load(f)
    with open(path + '/single_label_for_OnlineLR_CNY.pkl', "rb") as f:
        labels_ = pkl.load(f)

    train_x, train_y_A, train_y_N, train_y_Sigma = preprocess_training_data(num_steps, data_, labels_, train_list)

    val_x, val_y_A, val_y_N, val_y_Sigma = preprocess_training_data(num_steps, data_, labels_, val_list)

    test_x, test_y_A, test_y_N, test_y_Sigma = preprocess_training_data(num_steps, data_, labels_, test_list)

    rnn, output_states, label_A, label_N, label_Sigma, compare_A, compare_N, compare_Sigma, cost_A, cost_N, cost_Sigma, \
                                                 RRSE_A, RRSE_N, RRSE_Sigma, optimizer_A, optimizer_N, optimizer_Sigma  \
                                                            = Optimization(OnlineLR_cell, input_size, state_size,
                                                                           eta_max=eta_max, eta_min=eta_min, lr_1=lr_1, lr_2=lr_2,
                                                                           lr_3=lr_3, trainable=True, rd_seed=None)

    A_0, N_0, Sigma_0, E_epsilon_0, cov_epsilon_0, cell_state_0, \
                       hidden_state_0, y_t_pred_0, y_t_0, epsilon_t_0 = initialization(batch_size)

    val_A_0, val_N_0, val_Sigma_0, val_E_epsilon_0, val_cov_epsilon_0, val_cell_state_0, \
            val_hidden_state_0, val_y_t_pred_0, val_y_t_0, val_epsilon_t_0 = initialization(val_batch_size)

    with tf.Session() as sess:

        # number of batches for each epoch
        num_batches = len(train_x) // batch_size

        sess.run(tf.global_variables_initializer())

        train_loss_record_A = []
        train_loss_record_N = []
        train_loss_record_Sigma = []

        train_RRSE_record_A = []
        train_RRSE_record_N = []
        train_RRSE_record_Sigma = []

        val_loss_record_A = []
        val_loss_record_N = []
        val_loss_record_Sigma = []

        val_RRSE_record_A = []
        val_RRSE_record_N = []
        val_RRSE_record_Sigma = []

        for e_ in range(epochs):

            batches_train_loss_A = []
            batches_train_loss_N = []
            batches_train_loss_Sigma = []

            batches_train_RRSE_A = []
            batches_train_RRSE_N = []
            batches_train_RRSE_Sigma = []

            generator_ = FutureBatchGenerator(batch_size, shuffle_=False)

            for batch in range(num_batches):

                batch_train_x, batch_train_y_A, batch_train_y_N, batch_train_y_Sigma = \
                                            next(generator_.get_batches(train_x, train_y_A, train_y_N, train_y_Sigma))

                train_output, train_compare_A, train_compare_N, train_compare_Sigma, \
                                    train_cost_a, train_cost_n, train_cost_sig, RRSE_a, RRSE_n, RRSE_sig, _, _, _ = sess.run(
                                    [output_states, compare_A, compare_N, compare_Sigma, cost_A, cost_N, cost_Sigma,
                                     RRSE_A, RRSE_N, RRSE_Sigma, optimizer_A, optimizer_N, optimizer_Sigma],
                                                     feed_dict={
                                                                rnn.A_0: A_0,
                                                                rnn.N_0: N_0,
                                                                rnn.Sigma_0: Sigma_0,
                                                                rnn.E_epsilon_0: E_epsilon_0,
                                                                rnn.cov_epsilon_0: cov_epsilon_0,
                                                                rnn.cell_state_0: cell_state_0,
                                                                rnn.hidden_state_0: hidden_state_0,
                                                                rnn.y_t_0: y_t_0,
                                                                rnn.y_t_pred_0: y_t_pred_0,
                                                                rnn.epsilon_t_0: epsilon_t_0,
                                                                rnn._inputs: batch_train_x,
                                                                label_A: batch_train_y_A,
                                                                label_N: batch_train_y_N,
                                                                label_Sigma: batch_train_y_Sigma
                                                                })

                batches_train_loss_A.append(train_cost_a)
                batches_train_loss_N.append(train_cost_n)
                batches_train_loss_Sigma.append(train_cost_sig)

                batches_train_RRSE_A.append(RRSE_a)
                batches_train_RRSE_N.append(RRSE_n)
                batches_train_RRSE_Sigma.append(RRSE_sig)

            train_loss_record_A.append(np.mean(batches_train_loss_A))
            train_loss_record_N.append(np.mean(batches_train_loss_N))
            train_loss_record_Sigma.append(np.mean(batches_train_loss_Sigma))

            train_RRSE_record_A.append(np.mean(batches_train_RRSE_A))
            train_RRSE_record_N.append(np.mean(batches_train_RRSE_N))
            train_RRSE_record_Sigma.append(np.mean(batches_train_RRSE_Sigma))

            print("A_pred: {} \n".format(train_compare_A[:30]))
            # print("A_label: {} \n".format(batch_train_y_A[0]))
            print('cost_A: {}'.format(train_cost_a))
            print('RRSE_A: {}'.format(RRSE_a))
            print('\n')

            print("N_pred: {} \n".format(train_compare_N[:30]))
            # print("N_label: {} \n".format(batch_train_y_N[0]))
            print('cost_N: {}'.format(train_cost_n))
            print('RRSE_N: {}'.format(RRSE_n))
            print('\n')

            print('Sigma_pred: {} \n'.format(train_compare_Sigma[:30]))
            # print('Sigma_label: {} \n'.format(batch_train_y_Sigma[0]))
            print('cost_Sigma: {}'.format(train_cost_sig))
            print('RRSE_Sigma: {}'.format(RRSE_sig))
            print('\n')

            print('Epoch: {} Finished'.format(e_ + 1))
            print('\n\n')


            val_output, val_compare_A, val_compare_N, val_compare_Sigma, val_cost_a, val_cost_n, val_cost_sig, \
                    val_RRSE_a, val_RRSE_n, val_RRSE_sig, _, _, _ = sess.run([output_states, compare_A, compare_N, compare_Sigma,
                                                                               cost_A, cost_N, cost_Sigma, RRSE_A, RRSE_N, RRSE_Sigma,
                                                                               optimizer_A, optimizer_N, optimizer_Sigma],
                                                                     feed_dict={
                                                                                rnn.A_0: val_A_0,
                                                                                rnn.N_0: val_N_0,
                                                                                rnn.Sigma_0: val_Sigma_0,
                                                                                rnn.E_epsilon_0: val_E_epsilon_0,
                                                                                rnn.cov_epsilon_0: val_cov_epsilon_0,
                                                                                rnn.cell_state_0: val_cell_state_0,
                                                                                rnn.hidden_state_0: val_hidden_state_0,
                                                                                rnn.y_t_0: val_y_t_0,
                                                                                rnn.y_t_pred_0: val_y_t_pred_0,
                                                                                rnn.epsilon_t_0: val_epsilon_t_0,
                                                                                rnn._inputs: val_x,
                                                                                label_A: val_y_A,
                                                                                label_N: val_y_N,
                                                                                label_Sigma: val_y_Sigma
                                                                                })

            val_loss_record_A.append(val_cost_a)
            val_loss_record_N.append(val_cost_n)
            val_loss_record_Sigma.append(val_cost_sig)

            val_RRSE_record_A.append(val_RRSE_a)
            val_RRSE_record_N.append(val_RRSE_n)
            val_RRSE_record_Sigma.append(val_RRSE_sig)

            print("----------------------- validation results -----------------------")
            print("val_A_pred: {} \n".format(val_compare_A))
            # print("val_A_label: {} \n".format(val_y_A[0]))
            print('val_cost_A: {}'.format(val_cost_a))
            print('val_RRSE_A: {}'.format(val_RRSE_a))
            print('\n')

            print("val_N_pred: {} \n".format(val_compare_N))
            # print("val_N_label: {} \n".format(val_y_N[0]))
            print('val_cost_N: {}'.format(val_cost_n))
            print('val_RRSE_N: {}'.format(val_RRSE_n))
            print('\n')

            print('val_Sigma_pred: {} \n'.format(val_compare_Sigma))
            # print('val_Sigma_label: {} \n'.format(val_y_Sigma[0]))
            print('val_cost_Sigma: {} \n'.format(val_cost_sig))
            print('val_RRSE_Sigma: {}'.format(val_RRSE_sig))
            print("----------------------- validation finished -----------------------")
            print('\n\n')

        # plot
        # A
        with open(path+'/training_record_5Y.pkl') as f:
            pkl.dump([train_loss_record_A, train_RRSE_record_A, val_loss_record_A, val_RRSE_record_A,
                      train_loss_record_N, val_loss_record_N, train_RRSE_record_N, val_RRSE_record_N,
                      train_loss_record_Sigma, val_loss_record_Sigma, train_RRSE_record_Sigma, val_RRSE_record_Sigma], f)

        t_ = np.linspace(0, len(train_loss_record_A), num=len(train_loss_record_A))
        plt.figure()
        plt.title('loss A {}'.format(lr_1))
        plt.xlabel('Epochs')
        plt.ylabel('mean norm 2 over time')
        plt.plot(t_, train_loss_record_A, label='train_loss_A')
        plt.plot(t_, val_loss_record_A, label='val_loss_A')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title('RRSE A'.format(lr_1))
        plt.xlabel('Epochs')
        plt.ylabel('mean norm 2 over time')
        plt.plot(t_, train_RRSE_record_A, label='train_RRSE_A')
        plt.plot(t_, val_RRSE_record_A, label='val_RRSE_A')
        plt.legend()
        plt.grid()

        # N
        plt.figure()
        plt.title('loss N'.format(lr_2))
        plt.xlabel('Epochs')
        plt.ylabel('mean norm 2 over time')
        plt.plot(t_, train_loss_record_N, label='train_loss_N')
        plt.plot(t_, val_loss_record_N, label='val_loss_N')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title('RRSE N'.format(lr_2))
        plt.xlabel('Epochs')
        plt.ylabel('mean norm 2 over time')
        plt.plot(t_, train_RRSE_record_N, label='train_RRSE_N')
        plt.plot(t_, val_RRSE_record_N, label='val_RRSE_N')
        plt.legend()
        plt.grid()

        # Sigma
        plt.figure()
        plt.title('loss Sigma'.format(lr_3))
        plt.xlabel('Epochs')
        plt.ylabel('mean norm 2 over time')
        plt.plot(t_, train_loss_record_Sigma, label='train_loss_Sigma')
        plt.plot(t_, val_loss_record_Sigma, label='val_loss_Sigma')
        plt.legend()
        plt.grid()

        plt.figure()
        plt.title('RRSE Sigma'.format(lr_3))
        plt.xlabel('Epochs')
        plt.ylabel('mean norm 2 over time')
        plt.plot(t_, train_RRSE_record_Sigma, label='train_RRSE_Sigma')
        plt.plot(t_, val_RRSE_record_Sigma, label='val_RRSE_Sigma')
        plt.legend()
        plt.grid()

        plt.show()
       
