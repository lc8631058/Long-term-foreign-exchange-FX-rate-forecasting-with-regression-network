import time
import numpy as np
import tensorflow as tf
import pickle as pkl
from RegPred_Cell import RegCell_PredCell
from BayesianOptimization.BayesianOpt import BayesianOpt
from utils import brownian_motion_matrix

def idx_generator(idx_list, batch_size):
    len_ = len(idx_list)
    ii = 0

    while ii < len_:
        if ii + batch_size > len_:
            new_list = idx_list[ii:]
        else:
            new_list = idx_list[ii: ii + batch_size]
        ii += batch_size
        yield new_list

def initialization(batch_size, input_size, sigma_scale):
    """
    initialize the state
    sigma_scale: scale the sigma diagonal matrix
    """
    A_0 = np.zeros([batch_size, input_size, input_size])
    N_0 = np.zeros([batch_size, input_size, 1])
    Sigma_0 = np.array([np.eye(input_size)] * batch_size) * sigma_scale

    E_epsilon_0 = np.zeros([batch_size, input_size, 1])
    cov_epsilon_0 = np.zeros([batch_size, input_size, input_size])

    return A_0, N_0, Sigma_0, E_epsilon_0, cov_epsilon_0

def rebuild_chosen_initial_state(x_list, batch_size, input_size_1, input_size_2, input_size_3):
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
    A1_0 = np.zeros(shape=(batch_size, input_size_1, input_size_1))
    N1_0 = np.zeros(shape=(batch_size, input_size_1, 1))
    Sigma_10 = np.zeros(shape=(batch_size, input_size_1, input_size_1))
    eta_A1_t = np.zeros(shape=(batch_size, input_size_1, input_size_1))
    eta_N1_t = np.zeros(shape=(batch_size, input_size_1, 1))
    eta_Sigma_1t = np.zeros(shape=(batch_size, input_size_1, input_size_1))
    phi_1t = np.zeros(shape=(batch_size, input_size_1, 1))
    rho_1t = np.zeros(shape=(batch_size, input_size_1, input_size_1))

    #------------------------------------------------------------------#
    #                   Initial states for layer 2                     #
    #------------------------------------------------------------------#
    A2_0 = np.zeros(shape=(batch_size, input_size_2, input_size_2))
    N2_0 = np.zeros(shape=(batch_size, input_size_2, 1))
    Sigma_20 = np.zeros(shape=(batch_size, input_size_2, input_size_2))
    eta_A2_t = np.zeros(shape=(batch_size, input_size_2, input_size_2))
    eta_N2_t = np.zeros(shape=(batch_size, input_size_2, 1))
    eta_Sigma_2t = np.zeros(shape=(batch_size, input_size_2, input_size_2))
    phi_2t = np.zeros(shape=(batch_size, input_size_2, 1))
    rho_2t = np.zeros(shape=(batch_size, input_size_2, input_size_2))

    #------------------------------------------------------------------#
    #                  Initial states for layer 3                      #
    #------------------------------------------------------------------#
    eta_A3_t = np.zeros(shape=(batch_size, input_size_3, input_size_3))
    eta_N3_t = np.zeros(shape=(batch_size, input_size_3, 1))
    eta_Sigma_3t = np.zeros(shape=(batch_size, input_size_3, input_size_3))
    phi_3t = np.zeros(shape=(batch_size, input_size_3, 1))
    rho_3t = np.zeros(shape=(batch_size, input_size_3, input_size_3))

    for ii, x in enumerate(x_list):
        A1_0[ii] = x['A1_0']
        N1_0[ii] = x['N1_0']
        Sigma_10[ii] = x['Sigma_10']

        # np.tile copy into matrix
        eta_A1_t[ii] = np.tile(x['eta_A1_t'], (input_size_1, input_size_1))
        eta_N1_t[ii] = np.tile(x['eta_N1_t'], (input_size_1, 1))
        eta_Sigma_1t[ii] = np.tile(x['eta_Sigma_1t'], (input_size_1, input_size_1))
        phi_1t[ii] = np.tile(x['phi_1t'], (input_size_1, 1))
        rho_1t[ii] = np.tile(x['rho_1t'], (input_size_1, input_size_1))

        A2_0[ii] = np.array([[x['A2_0_11'], x['A2_0_12'], x['A2_0_13']],
                             [x['A2_0_21'], x['A2_0_22'], x['A2_0_23']],
                             [x['A2_0_31'], x['A2_0_32'], x['A2_0_33']]])

        N2_0[ii] = np.array([[x['N2_0_1']], [x['N2_0_2']], [x['N2_0_3']]])

        Sigma_20[ii] = np.array([[x['Sigma_20_11'], x['Sigma_20_12'], x['Sigma_20_13']],
                                 [x['Sigma_20_21'], x['Sigma_20_22'], x['Sigma_20_23']],
                                 [x['Sigma_20_31'], x['Sigma_20_32'], x['Sigma_20_33']]])

        # np.tile copy into matrix
        eta_A2_t[ii] = np.tile(x['eta_A2_t'], (input_size_2, input_size_2))
        eta_N2_t[ii] = np.tile(x['eta_N2_t'], (input_size_2, 1))
        eta_Sigma_2t[ii] = np.tile(x['eta_Sigma_2t'], (input_size_2, input_size_2))
        phi_2t[ii] = np.tile(x['phi_2t'], (input_size_2, 1))
        rho_2t[ii] = np.tile(x['rho_2t'], (input_size_2, input_size_2))

        eta_A3_t[ii] = np.tile(x['eta_A3_t'], (input_size_3, input_size_3))
        eta_N3_t[ii] = np.tile(x['eta_N3_t'], (input_size_3, 1))
        eta_Sigma_3t[ii] = np.tile(x['eta_Sigma_3t'], (input_size_3, input_size_3))
        phi_3t[ii] = np.tile(x['phi_3t'], (input_size_3, 1))
        rho_3t[ii] = np.tile(x['rho_3t'], (input_size_3, input_size_3))

    return A1_0, N1_0, Sigma_10, \
           eta_A1_t, eta_N1_t, eta_Sigma_1t, phi_1t, rho_1t, \
           A2_0, N2_0, Sigma_20, \
           eta_A2_t, eta_N2_t, eta_Sigma_2t, phi_2t, rho_2t, \
           eta_A3_t, eta_N3_t, eta_Sigma_3t, phi_3t, rho_3t


def rebuild_chosen_initial_state_layerwise(x_list, batch_size, input_size, layer=1, if_ANSigma=True):
    A_ = np.zeros(shape=(batch_size, input_size, input_size))
    N_ = np.zeros(shape=(batch_size, input_size, 1))
    Sigma_ = np.zeros(shape=(batch_size, input_size, input_size))
    eta_A_t = np.zeros(shape=(batch_size, input_size, input_size))
    eta_N_t = np.zeros(shape=(batch_size, input_size, 1))
    eta_Sigma_t = np.zeros(shape=(batch_size, input_size, input_size))
    phi_t = np.zeros(shape=(batch_size, input_size, 1))
    rho_t = np.zeros(shape=(batch_size, input_size, input_size))

    for ii, x in enumerate(x_list):

        eta_A_t[ii] = np.tile(x['eta_A{}_t'.format(layer)], (input_size, input_size))
        eta_N_t[ii] = np.tile(x['eta_N{}_t'.format(layer)], (input_size, 1))
        eta_Sigma_t[ii] = np.tile(x['eta_Sigma_{}t'.format(layer)], (input_size, input_size))
        phi_t[ii] = np.tile(x['phi_{}t'.format(layer)], (input_size, 1))
        rho_t[ii] = np.tile(x['rho_{}t'.format(layer)], (input_size, input_size))

        if input_size == 1 and layer == 1:
            A_[ii] = np.tile(x['A{}_0'.format(layer)], (input_size, input_size))
            N_[ii] = np.tile(x['N{}_0'.format(layer)], (input_size, 1))
            Sigma_[ii] = np.tile(x['Sigma_{}0'.format(layer)], (input_size, input_size))
            # return A_, N_, Sigma_, eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t

        elif input_size != 1 and layer != 1:

            # if include A,N,Sigma
            if if_ANSigma:

                # first dimension is for batch_size
                # A_ = np.ndarray((1, input_size, input_size), dtype=np.float32)
                # N_ = np.ndarray((1, input_size, 1), dtype=np.float32)
                # Sigma_ = np.ndarray((1, input_size, input_size), dtype=np.float32)

                for i_ in range(1, input_size+1):
                    N_[ii, i_-1] = x['N{}_0_{}'.format(layer, i_)]
                    for j_ in range(1, input_size+1):
                        A_[ii, i_-1, j_-1] = x['A{}_0_{}_{}'.format(layer, i_, j_)]
                        Sigma_[ii, i_-1, j_-1] = x['Sigma_{}0_{}_{}'.format(layer, i_, j_)]

                # A_ = np.tile(A_, (batch_size, 1, 1))
                # N_ = np.tile(N_, (batch_size, 1, 1))
                # Sigma_ = np.tile(Sigma_, (batch_size, 1, 1))

    if if_ANSigma:
        return A_, N_, Sigma_, eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t
    else:
        return eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t

def cal_mean_fx(BayesianOptimizer_list):
    return np.mean([each_BayOpt.max['fx'] for each_BayOpt in BayesianOptimizer_list])

def build_graph(expand_steps, input_size_1, input_size_2, input_size_3,
                input_size_4, size_N_3, size_N_2, size_N_1, total_num_layers=3
                ):
    """

    :param expand_steps:
    :param input_size_1:
    :param input_size_2:
    :param input_size_3:
    :param input_size_4:
    :param size_N_3: size of parameter N in decode layer 3
    :param size_N_2: size of parameter N in decode layer 2
    :param size_N_1: size of parameter N in decode layer 1
    :param num_layers: number of layers in total
    :DirectPredFuture: if True, prediction should start from end of [input_series + series_for_LossCal],
                       if False, prediction should start from end of [input_series], and predict series of length
                       [expand_steps_ForTrain + expand_steps_ForPred], and you only calculate the loss based on
                       part "expand_steps_ForPred".
    :return:
    """
    # create Graph
    g = tf.Graph()
    with g.as_default():

        # !!!!!!!!!!!!! Be careful !!!!!!!!!!!
        # make sure you feed label_y with size [time steps, batch_size, input_size]
        # If the first position is batch_size, then the network will use tf.scan to expand only
        # batch_size time steps instead of real time steps you want it does during decoding
        label_y = tf.placeholder(tf.float32, [None, None, input_size_1])

        num_samples = tf.placeholder(tf.int32)

        # 3 RegPred_Cell
        RegPred_Cell_list = [RegCell_PredCell(input_size_1, input_size_2, input_size_3, input_size_4,
                                        size_N_1, size_N_2, size_N_3, num_layer=idx) for idx in range(1, total_num_layers+1)]


        # encode_states[-1][-1]: the last time step in ZK with size (batch_size, input_size_4, 1)
        encode_states_list = [RegPred_Cell_.get_encode_states() for RegPred_Cell_ in RegPred_Cell_list]

        # the last layer's output, expand it identically for expand_steps
        # (expand_steps, num_samples, batch_size, input_size_4, 1)]
        ZK_list = [tf.tile(tf.expand_dims(tf.expand_dims(encode_states_[-1][-1], axis=0), axis=0),
                          [expand_steps, num_samples, 1, 1, 1]) for encode_states_ in encode_states_list]

        # Z0_T, Z1_T, Z2_T
        # for each: (num_samples, batch_size, input_size, 1)
        Decoder_initial_state_1 = tf.tile(tf.expand_dims(encode_states_list[0][-2][-1], axis=0), (num_samples, 1, 1, 1))

        Decoder_initial_state_2 = [tf.tile(tf.expand_dims(encode_states_list[1][5][-1], axis=0), (num_samples, 1, 1, 1)),
                                   tf.tile(tf.expand_dims(encode_states_list[1][-2][-1], axis=0), (num_samples, 1, 1, 1))
                                   ]

        Decoder_initial_state_3 = [tf.tile(tf.expand_dims(encode_states_list[2][5][-1], axis=0), (num_samples, 1, 1, 1)),
                                   tf.tile(tf.expand_dims(encode_states_list[2][11][-1], axis=0), (num_samples, 1, 1, 1)),
                                   tf.tile(tf.expand_dims(encode_states_list[2][-2][-1], axis=0), (num_samples, 1, 1, 1))
                                   ]

        Decoder_initial_state_list = [Decoder_initial_state_1, Decoder_initial_state_2, Decoder_initial_state_3]

        # ally ZK and random noises, encode_states[-1][-1]: Z3_t with size [batch, input_size_4, 1]
        # ZK_and_dW = (ZK, RegPred_Cell.dW_0, RegPred_Cell.dW_1, RegPred_Cell.dW_2)
        ZK_and_dW_1 = (ZK_list[0], RegPred_Cell_list[0].dW_0)
        ZK_and_dW_2 = (ZK_list[1], RegPred_Cell_list[1].dW_0, RegPred_Cell_list[1].dW_1)
        ZK_and_dW_3 = (ZK_list[2], RegPred_Cell_list[2].dW_0, RegPred_Cell_list[2].dW_1)
        ZK_and_dW_list = [ZK_and_dW_1, ZK_and_dW_2, ZK_and_dW_3]

        # get decode states for prediction, which are: Z0_T, Z1_T, Z2_T
        # decode_states_WithSigma = RegPred_Cell.get_decode_ForPred_states(Decoder_initial_state, ZK_and_dW)
        decode_states_WithSigma_list = [RegPred_Cell_list[idx].get_decode_layerwise_states(Decoder_initial_state_list[idx], ZK_and_dW_list[idx])
                                                                 for idx in range(len(ZK_and_dW_list))]

        # get Z0_T with size (expand_steps, num_samples, batch_size, 1) from the decode_states list
        pred_y_WithSigma_1 = tf.exp(tf.squeeze(decode_states_WithSigma_list[0], axis=-1))
        pred_y_WithSigma_2 = tf.exp(tf.squeeze(decode_states_WithSigma_list[1][0], axis=-1))
        pred_y_WithSigma_3 = tf.exp(tf.squeeze(decode_states_WithSigma_list[2][0], axis=-1))
        pred_y_WithSigma_list = [pred_y_WithSigma_1, pred_y_WithSigma_2, pred_y_WithSigma_3]

        # calculate mean and variance along 2nd dimension "num_samples"
        # (expand_steps, num_samples, batch_size, 1)
        # input should be train_x_ForPred, label should be train_y
        mean_from_samples_1, var_from_samples_1 = tf.nn.moments(pred_y_WithSigma_list[0], axes=1)
        mean_from_samples_2, var_from_samples_2 = tf.nn.moments(pred_y_WithSigma_list[1], axes=1)
        mean_from_samples_3, var_from_samples_3 = tf.nn.moments(pred_y_WithSigma_list[2], axes=1)

        mean_from_samples_list = [mean_from_samples_1, mean_from_samples_2, mean_from_samples_3]
        var_from_samples_list = [var_from_samples_1, var_from_samples_2, var_from_samples_3]

        # calculate the loss for mean and variance separately, then sum them up
        loss_mean_list = [tf.squeeze(tf.sqrt(tf.reduce_mean(tf.square(label_y - mean_from_samples_), axis=0)))
                                                 for mean_from_samples_ in mean_from_samples_list]

        loss_var_list = [tf.squeeze(tf.sqrt(tf.reduce_mean(tf.square(
                                    tf.square(label_y - mean_from_samples_) - var_from_samples_), axis=0)))
                                        for mean_from_samples_, var_from_samples_ in
                                            zip(mean_from_samples_list, var_from_samples_list)]

        # The total loss
        total_loss_list = [loss_mean_ + loss_var_ for loss_mean_, loss_var_ in zip(loss_mean_list, loss_var_list)]

    return g, num_samples, label_y, \
           RegPred_Cell_list, \
           pred_y_WithSigma_list, \
           total_loss_list, \
           loss_mean_list, \
           loss_var_list, \
           encode_states_list, \
           ZK_list, \
           Decoder_initial_state_list, \
           decode_states_WithSigma_list


def run_sess_with_graph(cell=None, BayesianOpt=None, acq=None, label=None,
                        batch_size=None, n_samples=None, sample_times=None,
                        dt=None, RandomState=None, expand_steps=None,
                        loss_mean=None, loss_var=None, loss=None,
                        pred_y=None, encode_states=None, ZK=None,
                        de_init=None, de_states=None,
                        input_size_1=None, input_size_2=None, input_size_3=None,
                        graph=None, BayesianOpt_iters=100, scipy_opt='L-BFGS-B',
                        num_warmup=1000, num_iters_ForAcqOpt=300, num_layer=None,
                        Z0_0=None, Z1_0=None, Z2_0=None, Z3_0=None,
                        E_epsilon_10=None, E_epsilon_20=None, E_epsilon_30=None,
                        cov_epsilon_10=None, cov_epsilon_20=None, cov_epsilon_30=None,
                        A1_0=None, A2_0=None, A3_0=None,
                        N1_0=None, N2_0=None, N3_0=None,
                        Sigma_10=None, Sigma_20=None, Sigma_30=None,
                        eta_A1_t=None, eta_A2_t=None, eta_A3_t=None,
                        eta_N1_t=None, eta_N2_t=None, eta_N3_t=None,
                        eta_Sigma_1t=None, eta_Sigma_2t=None, eta_Sigma_3t=None,
                        phi_1t=None, phi_2t=None, phi_3t=None,
                        rho_1t=None, rho_2t=None, rho_3t=None,
                        x_=None, y_=None, is_train=True, E2E_train=False):

    # if in test model, we don't iteratively find the best x_next
    if not is_train:
        BayesianOpt_iters = 1

    with tf.Session(graph=graph) as sess:

        for iter in range(1, BayesianOpt_iters + 1):

            # generate noises
            # size = (expand_steps, num_noise, batch_size, input_size, 1)
            if num_layer == 1 or num_layer == 2 or num_layer == 3:
                dW_0 = brownian_motion_matrix(dt, RandomState, expand_steps=expand_steps,
                                              num_noise=sample_times, batch_size=batch_size,
                                              input_size=input_size_1)

            if num_layer == 2 or num_layer == 3:
                dW_1 = brownian_motion_matrix(dt, RandomState, expand_steps=expand_steps,
                                              num_noise=sample_times, batch_size=batch_size,
                                              input_size=input_size_2)

            # if num_layer == 3:
            #     dW_2 = brownian_motion_matrix(dt, RandomState, expand_steps=expand_steps,
            #                                   num_noise=sample_times, batch_size=batch_size,
            #                                   input_size=input_size_3)

            if is_train:
                x_next_list = [each_BayOpt.find_next_x(acq,
                                                       opt_method=scipy_opt,
                                                       n_warmup=num_warmup,
                                                       n_iters=num_iters_ForAcqOpt)
                                                       for each_BayOpt in BayesianOpt]

            # sess.run(tf.global_variables_initializer())

            if num_layer == 1:

                if is_train:
                    A1_0, N1_0, Sigma_10, eta_A1_t, eta_N1_t, eta_Sigma_1t, phi_1t, rho_1t \
                        = rebuild_chosen_initial_state_layerwise(x_next_list, batch_size, input_size_1, layer=1, if_ANSigma=True)
                    Z1_0 = np.concatenate((A1_0, N1_0, Sigma_10), axis=1)

                feed_ = {
                    n_samples: sample_times,
                    cell.Z0_0: Z0_0,
                    cell._inputs: x_,
                    label: y_,
                    cell.Z1_0: Z1_0,
                    cell.A1_0: A1_0,
                    cell.N1_0: N1_0,
                    cell.Sigma_10: Sigma_10,
                    cell.E_epsilon_10: E_epsilon_10,
                    cell.cov_epsilon_10: cov_epsilon_10,
                    cell.dW_0: dW_0,

                    cell.eta_A1_t: eta_A1_t,
                    cell.eta_N1_t: eta_N1_t,
                    cell.eta_Sigma_1t: eta_Sigma_1t,
                    cell.phi_1t: phi_1t,
                    cell.rho_1t: rho_1t
                }

            elif num_layer == 2:

                if is_train:
                    A2_0, N2_0, Sigma_20, eta_A2_t, eta_N2_t, eta_Sigma_2t, phi_2t, rho_2t \
                        = rebuild_chosen_initial_state_layerwise(x_next_list, batch_size, input_size_2, layer=2, if_ANSigma=True)

                    Z2_0 = np.expand_dims(
                        np.array([np.concatenate([A2_0[idx].flatten(), N2_0[idx].flatten(), Sigma_20[idx].flatten()])
                                                                          for idx in range(A2_0.shape[0])]), axis=-1)

                    # Z2_0 = np.expand_dims(
                    #     np.expand_dims(np.concatenate((A2_0.flatten(), N2_0.flatten(), Sigma_20.flatten())), axis=0), axis=-1)

                feed_ = {
                    n_samples: sample_times,
                    cell.Z0_0: Z0_0,
                    cell._inputs: x_,
                    label: y_,
                    cell.Z1_0: Z1_0,
                    cell.A1_0: A1_0,
                    cell.N1_0: N1_0,
                    cell.Sigma_10: Sigma_10,
                    cell.E_epsilon_10: E_epsilon_10,
                    cell.cov_epsilon_10: cov_epsilon_10,
                    cell.dW_0: dW_0,

                    cell.Z2_0: Z2_0,
                    cell.A2_0: A2_0,
                    cell.N2_0: N2_0,
                    cell.Sigma_20: Sigma_20,
                    cell.E_epsilon_20: E_epsilon_20,
                    cell.cov_epsilon_20: cov_epsilon_20,
                    cell.dW_1: dW_1,

                    cell.eta_A1_t: eta_A1_t,
                    cell.eta_N1_t: eta_N1_t,
                    cell.eta_Sigma_1t: eta_Sigma_1t,
                    cell.phi_1t: phi_1t,
                    cell.rho_1t: rho_1t,

                    cell.eta_A2_t: eta_A2_t,
                    cell.eta_N2_t: eta_N2_t,
                    cell.eta_Sigma_2t: eta_Sigma_2t,
                    cell.phi_2t: phi_2t,
                    cell.rho_2t: rho_2t
                }

            elif num_layer == 3:

                if is_train:

                    if E2E_train: # if train end 2 end, means only train 3 layer structure, not layer-wise
                        A1_0, N1_0, Sigma_10, eta_A1_t, eta_N1_t, eta_Sigma_1t, phi_1t, rho_1t \
                            = rebuild_chosen_initial_state_layerwise(x_next_list, batch_size, input_size_1, layer=1,
                                                                     if_ANSigma=True)

                        A2_0, N2_0, Sigma_20, eta_A2_t, eta_N2_t, eta_Sigma_2t, phi_2t, rho_2t \
                            = rebuild_chosen_initial_state_layerwise(x_next_list, batch_size, input_size_2, layer=2,
                                                                     if_ANSigma=True)

                        eta_A3_t, eta_N3_t, eta_Sigma_3t, phi_3t, rho_3t \
                            = rebuild_chosen_initial_state_layerwise(x_next_list, batch_size, input_size_3, layer=3,
                                                                     if_ANSigma=False)
                    else:
                        eta_A3_t, eta_N3_t, eta_Sigma_3t, phi_3t, rho_3t \
                            = rebuild_chosen_initial_state_layerwise(x_next_list, batch_size, input_size_3, layer=3,
                                                                     if_ANSigma=False)

                    Z3_0 = np.expand_dims([np.concatenate([A3_0[idx].flatten(), N3_0[idx].flatten(), Sigma_30[idx].flatten()])
                                           for idx in range(A3_0.shape[0])], axis=-1)

                    # Z3_0 = np.expand_dims(
                    #     np.expand_dims(np.concatenate((A3_0.flatten(), N3_0.flatten(), Sigma_30.flatten())), axis=0),
                    #     axis=-1)

                feed_ = {
                    n_samples: sample_times,
                    cell.Z0_0: Z0_0,
                    cell._inputs: x_,
                    label: y_,
                    cell.Z1_0: Z1_0,
                    cell.A1_0: A1_0,
                    cell.N1_0: N1_0,
                    cell.Sigma_10: Sigma_10,
                    cell.E_epsilon_10: E_epsilon_10,
                    cell.cov_epsilon_10: cov_epsilon_10,
                    cell.dW_0: dW_0,

                    cell.Z2_0: Z2_0,
                    cell.A2_0: A2_0,
                    cell.N2_0: N2_0,
                    cell.Sigma_20: Sigma_20,
                    cell.E_epsilon_20: E_epsilon_20,
                    cell.cov_epsilon_20: cov_epsilon_20,
                    cell.dW_1: dW_1,

                    cell.Z3_0: Z3_0,
                    cell.A3_0: A3_0,
                    cell.N3_0: N3_0,
                    cell.Sigma_30: Sigma_30,
                    cell.E_epsilon_30: E_epsilon_30,
                    cell.cov_epsilon_30: cov_epsilon_30,
                    # cell.dW_2: dW_2,

                    cell.eta_A1_t: eta_A1_t,
                    cell.eta_N1_t: eta_N1_t,
                    cell.eta_Sigma_1t: eta_Sigma_1t,
                    cell.phi_1t: phi_1t,
                    cell.rho_1t: rho_1t,

                    cell.eta_A2_t: eta_A2_t,
                    cell.eta_N2_t: eta_N2_t,
                    cell.eta_Sigma_2t: eta_Sigma_2t,
                    cell.phi_2t: phi_2t,
                    cell.rho_2t: rho_2t,

                    cell.eta_A3_t: eta_A3_t,
                    cell.eta_N3_t: eta_N3_t,
                    cell.eta_Sigma_3t: eta_Sigma_3t,
                    cell.phi_3t: phi_3t,
                    cell.rho_3t: rho_3t
                }
            else:
                raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

            # output_loss_mean, output_loss_var, output_loss_WithSigma, output_pred_y_ForTrain, \
            # ouput_encode_states, output_ZK, output_Decoder_initial_state, output_decode_states_WithSigma \
            #     = sess.run([loss_mean, loss_var, loss, pred_y,
            #                 encode_states, ZK, de_init, de_states],
            #                 feed_dict= feed_
            #                )
            output_loss_mean, output_loss_var, output_loss_WithSigma, output_pred_y_ForTrain \
                = sess.run([loss_mean, loss_var, loss, pred_y],
                           feed_dict=feed_
                           )

            if is_train:
                yield x_next_list, output_loss_mean, output_loss_var, output_loss_WithSigma, output_pred_y_ForTrain
            else:
                yield output_loss_mean, output_loss_var, output_loss_WithSigma, output_pred_y_ForTrain


def Train_RegPred_layerwise(train_x=None, train_y=None, n_layers=None, BayesianOpt_iters=None, RegPred_Cell_ForTrain=None,
                  acq_function=None, label_y_ForTrain=None, batch_size=None, num_samples_ForTrain=None,
                  sample_times_ForTrain=None, dt=None, RandomState_instance=None, expand_steps_ForTrain=None,
                  loss_mean_ForTrain=None, loss_var_ForTrain=None, loss_WithSigma_ForTrain=None, pred_y_WithSigma_ForTrain=None,
                  encode_states_ForTrain=None, ZK_ForTrain=None, Decoder_initial_state_ForTrain=None,
                  decode_states_WithSigma_ForTrain=None, g_ForTrain=None, scipy_opt=None, num_warmup=None, num_iters_ForAcqOpt=None,
                  input_size_1=None, input_size_2=None, input_size_3=None, Z0_0=None, Z1_0=None, Z2_0=None,
                  Z3_0=None, E_epsilon_10=None, E_epsilon_20=None, E_epsilon_30=None, cov_epsilon_10=None, cov_epsilon_20=None,
                  cov_epsilon_30=None, A1_0=None, A2_0=None, A3_0=None, N1_0=None, N2_0=None, N3_0=None, Sigma_10=None, Sigma_20=None,
                  Sigma_30=None, eta_A10=None, eta_A20=None, eta_N10=None, eta_N20=None, eta_Sigma_10=None, eta_Sigma_20=None,
                  phi_10=None, phi_20=None, rho_10=None, rho_20=None, bounds_=None, random_state=None, is_train=True):

    # record the time
    start_ = time.time()
    print('time count begin for {}st layer training'.format(n_layers))

    # build multiple BayesianOptimizer, same size as batch_size

    BayesianOptimizer_list = [BayesianOpt(obj_function=None, pbounds=bounds_,
                                          random_state=random_state,
                                          RegPred_layer_num=n_layers)
                              for idx in range(batch_size)]

    for iter in range(1, BayesianOpt_iters + 1):
        x_next_list, output_loss_mean_, output_loss_var_, output_total_loss_, output_pred_y_ForTrain_ \
            = next(run_sess_with_graph(cell=RegPred_Cell_ForTrain,
                                       BayesianOpt=BayesianOptimizer_list,
                                       acq=acq_function,
                                       label=label_y_ForTrain,
                                       batch_size=batch_size,
                                       n_samples=num_samples_ForTrain,
                                       sample_times=sample_times_ForTrain,
                                       dt=dt, RandomState=RandomState_instance,
                                       expand_steps=expand_steps_ForTrain,
                                       loss_mean=loss_mean_ForTrain,
                                       loss_var=loss_var_ForTrain,
                                       loss=loss_WithSigma_ForTrain,
                                       pred_y=pred_y_WithSigma_ForTrain,
                                       encode_states=encode_states_ForTrain,
                                       ZK=ZK_ForTrain,
                                       de_init=Decoder_initial_state_ForTrain,
                                       de_states=decode_states_WithSigma_ForTrain,
                                       graph=g_ForTrain,
                                       input_size_1=input_size_1,
                                       input_size_2=input_size_2,
                                       input_size_3=input_size_3,
                                       BayesianOpt_iters=BayesianOpt_iters,
                                       scipy_opt=scipy_opt,
                                       num_warmup=num_warmup,
                                       num_iters_ForAcqOpt=num_iters_ForAcqOpt,
                                       num_layer=n_layers,
                                       Z0_0=Z0_0, Z1_0=Z1_0, Z2_0=Z2_0, Z3_0=Z3_0,
                                       E_epsilon_10=E_epsilon_10,
                                       E_epsilon_20=E_epsilon_20,
                                       E_epsilon_30=E_epsilon_30,
                                       cov_epsilon_10=cov_epsilon_10,
                                       cov_epsilon_20=cov_epsilon_20,
                                       cov_epsilon_30=cov_epsilon_30,
                                       A1_0=A1_0, A2_0=A2_0, A3_0=A3_0,
                                       N1_0=N1_0, N2_0=N2_0, N3_0=N3_0,
                                       Sigma_10=Sigma_10, Sigma_20=Sigma_20, Sigma_30=Sigma_30,
                                       eta_A1_t=eta_A10, eta_A2_t=eta_A20,
                                       eta_N1_t=eta_N10, eta_N2_t=eta_N20,
                                       eta_Sigma_1t=eta_Sigma_10, eta_Sigma_2t=eta_Sigma_20,
                                       phi_1t=phi_10, phi_2t=phi_20,
                                       rho_1t=rho_10, rho_2t=rho_20,
                                       x_=train_x,
                                       y_=train_y,
                                       is_train=is_train
                                       )
                   )

        # if f(x) is nan or inf, assign a super small negative value to it
        # detect inf and nan from results
        output_cost_next_ = - output_total_loss_

        output_cost_next_inf = np.isinf(output_total_loss_).astype(int)
        output_cost_next_nan = np.isnan(output_total_loss_).astype(int)

        # to minimize f(x) == to maximize the minus f(x)
        for ii, (inf_nan_) in enumerate(output_cost_next_inf+output_cost_next_nan):
            if inf_nan_==1:
                output_cost_next_[ii] = -1e+30

        # generate a BayesianOptimizer for each fx evaluated from the network
        for ii, (x_next_, each_BayOpt) in enumerate(zip(x_next_list, BayesianOptimizer_list)):
            each_BayOpt.add_point(x_next_, output_cost_next_[ii])

        print("iters: {}, ".format(iter), "loss_mean: {0:.4f}".format(np.mean(output_loss_mean_)),
              "loss_var: {0:.4f}".format(np.mean(output_loss_var_)), "loss_total mean: {0:.4f}".format(np.mean(output_cost_next_)))

        if iter % 100 == 0:
            mean_best_fx = cal_mean_fx(BayesianOptimizer_list)
            print("\n \n")
            print("The optimum until now is fx: {} \n \n".format(mean_best_fx))

    end_ = time.time()

    return BayesianOptimizer_list, (end_ - start_) / 60