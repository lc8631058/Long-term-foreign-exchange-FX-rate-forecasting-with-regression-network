import tensorflow as tf
import numpy as np
import time

from utils import rebuild_chosen_initial_state_layerwise, brownian_motion_matrix

class RegCell_PredCell(object):

    def __init__(self, input_size_1, input_size_2, input_size_3, input_size_4,
                       size_N_1, size_N_2, size_N_3, num_layer=None, add_noise_to2nd=True, train=True):
        self.num_layer = num_layer

        # input_size for each layer in Encoder
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.input_size_3 = input_size_3
        self.input_size_4 = input_size_4 # input_size for the 4th layer

        # size of N in Decoder
        self.size_N_1 = size_N_1 # for the 1st decoder
        self.size_N_2 = size_N_2 # for the 2nd decoder
        self.size_N_3 = size_N_3 # for the 3rd decoder

        self.add_noise_to2nd = add_noise_to2nd

        # inputs [batch, num_steps, input_size]
        self.ori_inputs = tf.placeholder(tf.float32, [None, None, self.input_size_1], name='inputs_')

        # inputs [batch, num_steps, 2, input_size], where 2 represents the sequence of y_t_1 and y_t
        self.ext_inputs = tf.expand_dims(self.ori_inputs, dim=2)
        self._inputs = tf.concat([self.ext_inputs[:, :-1], self.ext_inputs[:, 1:]], axis=2)

        # sacn_inputs just switch the position 0 and 1 in inputs, in order to be compatible with the tf.scan function
        self.scan_inputs = tf.transpose(self._inputs, perm=[1, 0, 2, 3], name='scan_inputs')

        # create A, N, Sigma for each layer
        self.A1_0, self.N1_0, self.Sigma_10, self.E_epsilon_10, self.cov_epsilon_10 = \
            self.parameter_initializer(self.input_size_1)

        self.A2_0, self.N2_0, self.Sigma_20, self.E_epsilon_20, self.cov_epsilon_20 = \
            self.parameter_initializer(self.input_size_2)

        self.A3_0, self.N3_0, self.Sigma_30, self.E_epsilon_30, self.cov_epsilon_30 = \
            self.parameter_initializer(self.input_size_3)

        # [num_noise, expand_steps, batch_size, input_size, input_size]
        if train:
            self.dW_0 = tf.placeholder(tf.float32, [None, None, None, None, self.input_size_1, 1], name='dW_0')
            self.dW_1 = tf.placeholder(tf.float32, [None, None, None, None, self.input_size_2, 1], name='dW_1')
        else:
            self.dW_0 = tf.placeholder(tf.float32, [None, None, None, self.input_size_1, 1], name='dW_0')
            self.dW_1 = tf.placeholder(tf.float32, [None, None, None, self.input_size_2, 1], name='dW_1')
        # self.dW_2 = tf.placeholder(tf.float32, [None, None, None, self.input_size_3, 1], name='dW_2')

        # [batch_size, inputfe_size, 1] the initial state of Z0, ..., Z4 for corresponding Encode layers
        self.Z0_0 = tf.placeholder(tf.float32, [None, self.input_size_1, 1], name='Z0_0')
        self.Z1_0 = tf.placeholder(tf.float32, [None, self.input_size_2, 1], name='Z1_0')
        self.Z2_0 = tf.placeholder(tf.float32, [None, self.input_size_3, 1], name='Z2_0')
        self.Z3_0 = tf.placeholder(tf.float32, [None, self.input_size_4, 1], name='Z3_0')

        # eta for each layer, the 5 learning rates
        self.eta_A1_t = tf.placeholder(tf.float32, [None, self.input_size_1, self.input_size_1], name='eta_A1_t')
        self.eta_N1_t = tf.placeholder(tf.float32, [None, self.input_size_1, 1], name='eta_N1_t')
        self.eta_Sigma_1t = tf.placeholder(tf.float32, [None, self.input_size_1, self.input_size_1], name='eta_Sigma_1t')
        self.phi_1t = tf.placeholder(tf.float32, [None, self.input_size_1, 1], name='phi_1t')
        self.rho_1t = tf.placeholder(tf.float32, [None, self.input_size_1, self.input_size_1], name='rho_1t')

        self.eta_A2_t = tf.placeholder(tf.float32, [None, self.input_size_2, self.input_size_2], name='eta_A2_t')
        self.eta_N2_t = tf.placeholder(tf.float32, [None, self.input_size_2, 1], name='eta_N2_t')
        self.eta_Sigma_2t = tf.placeholder(tf.float32, [None, self.input_size_2, self.input_size_2], name='eta_Sigma_2t')
        self.phi_2t = tf.placeholder(tf.float32, [None, self.input_size_2, 1], name='phi_2t')
        self.rho_2t = tf.placeholder(tf.float32, [None, self.input_size_2, self.input_size_2], name='rho_2t')

        self.eta_A3_t = tf.placeholder(tf.float32, [None, self.input_size_3, self.input_size_3], name='eta_A3_t')
        self.eta_N3_t = tf.placeholder(tf.float32, [None, self.input_size_3, 1], name='eta_N3_t')
        self.eta_Sigma_3t = tf.placeholder(tf.float32, [None, self.input_size_3, self.input_size_3], name='eta_Sigma_3t')
        self.phi_3t = tf.placeholder(tf.float32, [None, self.input_size_3, 1], name='phi_3t')
        self.rho_3t = tf.placeholder(tf.float32, [None, self.input_size_3, self.input_size_3], name='rho_3t')

        if self.num_layer == 1:
            self.Encoder_initial_state = [self.A1_0, self.N1_0, self.Sigma_10, self.E_epsilon_10, self.cov_epsilon_10, self.Z0_0, self.Z1_0]

        elif self.num_layer == 2:
            self.Encoder_initial_state = [self.A1_0, self.N1_0, self.Sigma_10, self.E_epsilon_10, self.cov_epsilon_10, self.Z0_0,
                                          self.A2_0, self.N2_0, self.Sigma_20, self.E_epsilon_20, self.cov_epsilon_20, self.Z1_0, self.Z2_0]
        elif self.num_layer == 3:
            self.Encoder_initial_state = [self.A1_0, self.N1_0, self.Sigma_10, self.E_epsilon_10, self.cov_epsilon_10, self.Z0_0,
                                          self.A2_0, self.N2_0, self.Sigma_20, self.E_epsilon_20, self.cov_epsilon_20, self.Z1_0,
                                          self.A3_0, self.N3_0, self.Sigma_30, self.E_epsilon_30, self.cov_epsilon_30, self.Z2_0, self.Z3_0]
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

    def parameter_initializer(self, input_size):
        """
        initialize the states for tf.scan
        :param input_size: input size
        :return: initialized states
        """
        A_0 = tf.placeholder(tf.float32, [None, input_size, input_size])
        N_0 = tf.placeholder(tf.float32, [None, input_size, 1])
        Sigma_0 = tf.placeholder(tf.float32, [None, input_size, input_size])
        E_epsilon_0 = tf.placeholder(tf.float32, [None, input_size, 1])
        cov_epsilon_0 = tf.placeholder(tf.float32, [None, input_size, input_size])

        return A_0, N_0, Sigma_0, E_epsilon_0, cov_epsilon_0

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
        # big T indicates transpose
        epsilon_t_T = tf.transpose(epsilon_t, perm=[0, 2, 1])
        E_epsilon_t_T = tf.transpose(E_epsilon_t, perm=[0, 2, 1])

        EMA_cov_epsilon = tf.multiply(rho, (tf.matmul(epsilon_t, epsilon_t_T) - tf.matmul(epsilon_t, E_epsilon_t_T)
                                            - tf.matmul(E_epsilon_t, epsilon_t_T) + tf.matmul(E_epsilon_t,
                                                                                              E_epsilon_t_T))) \
                          + tf.multiply((1. - rho), previous_EMA_cov_epsilon)

        return EMA_cov_epsilon

    def RegCell_updating(self, Zk_1_t, Zk_1_t_1, Ak_t_1, Nk_t_1, Sigma_kt_1, E_epsilon_kt_1, cov_epsilon_kt_1,
                               eta_Ak_t, eta_Nk_t, eta_Sigma_kt, phi_kt, rho_kt):
        """
        the operations in RegCell
        :param Zk_1_t: Z^(k_1)_t
        :param Zk_1_t_1: Z^(k_1)_t-1
        :param Ak_t_1: A^(k)_t-1
        :param Nk_t_1: N^(k)_t-1
        :param Sigma_kt_1: Sigma^(k)_t-1
        :param E_epsilon_kt_1: E_epsilon^(k)_t-1
        :param cov_epsilon_kt_1: cov_epsilon^(k)_t-1
        :param eta_Ak_t: eta_A^(k)_t
        :param eta_Nk_t: eta_N^(k)_t
        :param eta_Sigma_kt: eta_Sigma^(k)_t
        :param phi_kt: phi^(k)_t
        :param rho_kt: rho^(k)_t
        :return: Ak_t, Nk_t, Sigma_kt, E_epsilon_kt, cov_epsilon_kt
        """
        # transpose [batch, input_size_1, 1] -> [batch, 1, input_size_1]
        # big T means transpose
        Zk_1_tT = tf.transpose(Zk_1_t, perm=[0, 2, 1])
        Sigma_kt_1T = tf.transpose(Sigma_kt_1, perm=[0, 2, 1])

        delta_Zk_t = Zk_1_t - Zk_1_t_1

        epsilon_kt = delta_Zk_t - (tf.matmul(Ak_t_1, Zk_1_t_1) + Nk_t_1)

        Ak_t = Ak_t_1 + 2. * eta_Ak_t * tf.matmul(epsilon_kt, Zk_1_tT)

        Nk_t = Nk_t_1 + 2. * eta_Nk_t * epsilon_kt

        E_epsilon_kt = self.EMA_epsilon(E_epsilon_kt_1, epsilon_kt, phi_kt)

        cov_epsilon_kt = self.EMA_cov_epsilon(cov_epsilon_kt_1, epsilon_kt, E_epsilon_kt, rho_kt)

        Sigma_kt = Sigma_kt_1 - 4. * tf.multiply(eta_Sigma_kt, tf.matmul(
                   (tf.matmul(Sigma_kt_1, Sigma_kt_1T) - cov_epsilon_kt), Sigma_kt_1))

        return Ak_t, Nk_t, Sigma_kt, E_epsilon_kt, cov_epsilon_kt

    def Encode(self, previous_states, Z):

        # Z3_t_1: ZK
        if self.num_layer == 1:
            A1_t_1, N1_t_1, Sigma_1t_1, E_epsilon_1t_1, cov_epsilon_1t_1, Z0_t_1, Z1_t_1 = previous_states

        elif self.num_layer == 2:
            A1_t_1, N1_t_1, Sigma_1t_1, E_epsilon_1t_1, cov_epsilon_1t_1, Z0_t_1, \
            A2_t_1, N2_t_1, Sigma_2t_1, E_epsilon_2t_1, cov_epsilon_2t_1, Z1_t_1, Z2_t_1 = previous_states

        elif self.num_layer == 3:
            A1_t_1, N1_t_1, Sigma_1t_1, E_epsilon_1t_1, cov_epsilon_1t_1, Z0_t_1, \
            A2_t_1, N2_t_1, Sigma_2t_1, E_epsilon_2t_1, cov_epsilon_2t_1, Z1_t_1, \
            A3_t_1, N3_t_1, Sigma_3t_1, E_epsilon_3t_1, cov_epsilon_3t_1, Z2_t_1, Z3_t_1 = previous_states
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

        # ---------------------------------------------------------------------- #
        #                      Processing 1st encode layer
        # ---------------------------------------------------------------------- #
        # For each parameter, the first number is layer number (k), the second number is time index,
        # e.g. Z0_t_1 == Z^(0)_t-1, epsilon_1t == epsilon^(1)_t
        # unpack the input y into [batch, Z0_t-1] and [batch, Z0_t]
        # Z0_t = Y_t
        Z0_t_1, Z0_t = tf.unstack(Z, axis=1)

        # for matrix multiplication; 0 means layer (0)
        Z0_t_1 = tf.expand_dims(Z0_t_1, axis=-1)
        Z0_t = tf.expand_dims(Z0_t, axis=-1)

        # Learning rate for updating A,N,Sigma; 1 means layer (1)
        A1_t, N1_t, Sigma_1t, E_epsilon_1t, cov_epsilon_1t = \
            self.RegCell_updating(Z0_t, Z0_t_1, A1_t_1, N1_t_1, Sigma_1t_1, E_epsilon_1t_1, cov_epsilon_1t_1,
                                  self.eta_A1_t, self.eta_N1_t, self.eta_Sigma_1t, self.phi_1t, self.rho_1t)

        # construct Z1_t
        Z1_t = tf.concat([A1_t, N1_t, Sigma_1t], axis=1)

        # if just 1 layer, return the result
        if self.num_layer == 1:
            return [A1_t, N1_t, Sigma_1t, E_epsilon_1t, cov_epsilon_1t, Z0_t, Z1_t]

        # ---------------------------------------------------------------------- #
        #                      Processing 2nd encode layer
        # ---------------------------------------------------------------------- #
        if self.num_layer == 2 or self.num_layer == 3:
            A2_t, N2_t, Sigma_2t, E_epsilon_2t, cov_epsilon_2t = \
                self.RegCell_updating(Z1_t, Z1_t_1, A2_t_1, N2_t_1, Sigma_2t_1, E_epsilon_2t_1,
                                      cov_epsilon_2t_1, self.eta_A2_t, self.eta_N2_t, self.eta_Sigma_2t, self.phi_2t, self.rho_2t)

            # construct Z2_t
            A2_t_fla = tf.reshape(A2_t, [tf.shape(A2_t)[0], tf.square(self.size_N_2), 1])
            N2_t_fla = tf.reshape(N2_t, [tf.shape(N2_t)[0], self.size_N_2, 1])
            Sigma_2t_fla = tf.reshape(Sigma_2t, [tf.shape(Sigma_2t)[0], tf.square(self.size_N_2), 1])

            Z2_t = tf.concat([A2_t_fla, N2_t_fla, Sigma_2t_fla], axis=1)

        if self.num_layer == 2:
            return [A1_t, N1_t, Sigma_1t, E_epsilon_1t, cov_epsilon_1t, Z0_t,
                    A2_t, N2_t, Sigma_2t, E_epsilon_2t, cov_epsilon_2t, Z1_t, Z2_t]

        # ---------------------------------------------------------------------- #
        #                      Processing 3rd encode layer
        # ---------------------------------------------------------------------- #

        A3_t, N3_t, Sigma_3t, E_epsilon_3t, cov_epsilon_3t = \
            self.RegCell_updating(Z2_t, Z2_t_1, A3_t_1, N3_t_1, Sigma_3t_1, E_epsilon_3t_1,
                                  cov_epsilon_3t_1, self.eta_A3_t, self.eta_N3_t, self.eta_Sigma_3t, self.phi_3t, self.rho_3t)

        # construct Z2_t
        A3_t_fla = tf.reshape(A3_t, [tf.shape(A3_t)[0], tf.square(self.size_N_3), 1])
        N3_t_fla = tf.reshape(N3_t, [tf.shape(N3_t)[0], self.size_N_3, 1])
        Sigma_3t_fla = tf.reshape(Sigma_3t, [tf.shape(Sigma_3t)[0], tf.square(self.size_N_3), 1])

        Z3_t = tf.concat([A3_t_fla, N3_t_fla, Sigma_3t_fla], axis=1)

        return [A1_t, N1_t, Sigma_1t, E_epsilon_1t, cov_epsilon_1t, Z0_t,
                A2_t, N2_t, Sigma_2t, E_epsilon_2t, cov_epsilon_2t, Z1_t,
                A3_t, N3_t, Sigma_3t, E_epsilon_3t, cov_epsilon_3t, Z2_t, Z3_t]

    def Decode_layerwise(self, previous_state, ZK_and_dW):
        """
        Decode process for prediction with random noise.
        :param previous_state: Z0_T, Z1_T, Z2_T, they are initial values for Decoder layer 0, 1 and 2.
        :param ZK_and_dW: a tuple, contains Z^K_t and dW^K-1_t, dW^K-2_t, dW^K-3_t (dW_2t, dW_1t, dW_0t).
                                   The first is the A, N, Sigma value from layer K with size
                                   [expand_steps, batch, input_size, 1], which is the last layer of Encoder.
                                   The second is the brownian motion for each time step of the K-1 (2) Decoder layer
                                   with size [expand_steps, batch, input_size, input_size].
                                   The third is the brownian motion for each time step of the K-2 (1) Decoder layer.
                                   The fourth is the brownian motion for each time step of the K-3 (0) Decoder layer
        :return: decoded results for Decoder layer 0, 1, 2, which layer 0 outputs the FX rates.
        """
        # big T means the total time steps of Encoder, which means the start time step of Decoder
        # [num_samples, batch_size, input_size, 1]
        if self.num_layer == 1:
            Z0_T = previous_state
        elif self.num_layer == 2:
            Z0_T, Z1_T = previous_state
        elif self.num_layer == 3:
            Z0_T, Z1_T, Z2_T = previous_state
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

        # deconstruct ZK_and_dW into ZK and dW separately
        # [num_samples, batch_size, input_size, 1]
        ZK = ZK_and_dW[0]

        #----------------------------------------------------------------------#
        #                      Processing 3rd decode layer
        #----------------------------------------------------------------------#
        if self.num_layer == 3:
            # noises
            dW_0T = ZK_and_dW[1]  # the second one is for the Decode layer 0
            dW_1T = ZK_and_dW[2]  # the third one is for the Decode layer 1
            # dW_2T = ZK_and_dW[3]  # the fourth one is for the Decode layer 2

            # Deconstruct Z^(K) into A^(K), N^(K), Sigma^(K)
            A3_T = ZK[:, :, :tf.square(self.size_N_3)]
            A3_T_mat = tf.reshape(A3_T, [tf.shape(A3_T)[0], tf.shape(A3_T)[1], self.size_N_3, self.size_N_3]) # To matrix

            N3_T = ZK[:, :, tf.square(self.size_N_3):tf.square(self.size_N_3) + self.size_N_3]

            Sigma_3T = ZK[:, :, tf.square(self.size_N_3) + self.size_N_3:]
            Sigma_3T_mat = tf.reshape(Sigma_3T, [tf.shape(Sigma_3T)[0], tf.shape(Sigma_3T)[1], self.size_N_3, self.size_N_3])  # To matrix

            # [num_samples, input_size_3, 1] + [num_samples, input_size_3,input_size_3] x [num_samples, input_size_3, 1]
            # + [num_samples, input_size_3, 1] + [num_samples, input_size_3,input_size_3] x [num_samples, input_size_3, 1]
            # Z2_T1 = Z2_T + tf.matmul(A3_T_mat, Z2_T) + N3_T + tf.matmul(Sigma_3T_mat, dW_2T)
            Z2_T1 = Z2_T + tf.matmul(A3_T_mat, Z2_T) + N3_T
            # Z2_T1 = self.scaled_tanh(Z2_T1, self.m2)
            A2_T = Z2_T[:, :, :tf.square(self.size_N_2)]
            N2_T = Z2_T[:, :, tf.square(self.size_N_2):tf.square(self.size_N_2) + self.size_N_2]
            Sigma_2T = Z2_T[:, :, tf.square(self.size_N_2) + self.size_N_2:]

        #----------------------------------------------------------------------#
        #                      Processing 2nd decode layer
        #----------------------------------------------------------------------#
        if self.num_layer == 2:
            # noises
            dW_0T = ZK_and_dW[1]  # the second one is for the Decode layer 0
            dW_1T = ZK_and_dW[2]  # the third one is for the Decode layer 1

            A2_T = ZK[:, :, :tf.square(self.size_N_2)]
            N2_T = ZK[:, :, tf.square(self.size_N_2):tf.square(self.size_N_2) + self.size_N_2]
            Sigma_2T = ZK[:, :, tf.square(self.size_N_2) + self.size_N_2:]

        if self.num_layer == 3 or self.num_layer == 2:
            A2_T_mat = tf.reshape(A2_T, [tf.shape(A2_T)[0], tf.shape(A2_T)[1], self.size_N_2, self.size_N_2])  # To matrix

            Sigma_2T_mat = tf.reshape(Sigma_2T, [tf.shape(Sigma_2T)[0], tf.shape(Sigma_2T)[1], self.size_N_2, self.size_N_2])  # To matrix

            # [input_size_2, 1] + [input_size_2,input_size_2] x [input_size_2, 1] + [input_size_2, 1]
            # + [input_size_2,input_size_2] x [input_size_2, 1]
            # Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T + tf.matmul(Sigma_2T_mat, dW_1T)
            # Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T + tf.matmul(Sigma_2T_mat, dW_1T)
            if self.add_noise_to2nd:
                Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T + tf.matmul(Sigma_2T_mat, dW_1T)
            else:
                Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T
            # Z1_T1 = self.scaled_tanh(Z1_T1, self.m1)

            A1_T = Z1_T[:, :, :tf.square(self.size_N_1)]
            N1_T = Z1_T[:, :, tf.square(self.size_N_1):tf.square(self.size_N_1) + self.size_N_1]
            Sigma_1T = Z1_T[:, :, tf.square(self.size_N_1) + self.size_N_1:]

        # ---------------------------------------------------------------------- #
        #                      Processing 1st decode layer
        # ---------------------------------------------------------------------- #
        if self.num_layer == 1:
            # noises
            dW_0T = ZK_and_dW[1]  # the second one is for the Decode layer 0

            A1_T = ZK[:, :, :tf.square(self.size_N_1)]
            N1_T = ZK[:, :, tf.square(self.size_N_1):tf.square(self.size_N_1) + self.size_N_1]
            Sigma_1T = ZK[:, :, tf.square(self.size_N_1) + self.size_N_1:]

        if self.num_layer == 3 or self.num_layer == 2 or self.num_layer == 1:
            # [input_size_1, 1] + [input_size_1,input_size_1] x [input_size_1, 1] + [input_size_1, 1]
            # + [input_size_1,input_size_1] x [input_size_1, 1]
            Z0_T1 = Z0_T + tf.matmul(A1_T, Z0_T) + N1_T + tf.matmul(Sigma_1T, dW_0T)
            # Z0_T1 = 1. * tf.tanh((Z0_T + tf.matmul(A1_T, Z0_T) + N1_T) /1.)
            # Z0_T1 = self.scaled_tanh(Z0_T1, self.m0)
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

        if self.num_layer == 1:
            return Z0_T1
        elif self.num_layer == 2:
            return [Z0_T1, Z1_T1]
        elif self.num_layer == 3:
            # T1 means time T+1
            return [Z0_T1, Z1_T1, Z2_T1]
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

    def Decode_layerwise_AvgLoss(self, previous_state, ZK_and_dW):
        """
        Decode process for prediction with random noise.
        :param previous_state: Z0_T, Z1_T, Z2_T, they are initial values for Decoder layer 0, 1 and 2.
        :param ZK_and_dW: a tuple, contains Z^K_t and dW^K-1_t, dW^K-2_t, dW^K-3_t (dW_2t, dW_1t, dW_0t).
                                   The first is the A, N, Sigma value from layer K with size
                                   [expand_steps, batch, input_size, 1], which is the last layer of Encoder.
                                   The second is the brownian motion for each time step of the K-1 (2) Decoder layer
                                   with size [expand_steps, batch, input_size, input_size].
                                   The third is the brownian motion for each time step of the K-2 (1) Decoder layer.
                                   The fourth is the brownian motion for each time step of the K-3 (0) Decoder layer
        :return: decoded results for Decoder layer 0, 1, 2, which layer 0 outputs the FX rates.
        """
        # big T means the total time steps of Encoder, which means the start time step of Decoder
        # [num_samples, batch_size, input_size, 1]
        if self.num_layer == 1:
            Z0_T = previous_state
        elif self.num_layer == 2:
            Z0_T, Z1_T = previous_state
        elif self.num_layer == 3:
            Z0_T, Z1_T, Z2_T = previous_state
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

        # deconstruct ZK_and_dW into ZK and dW separately
        # [num_samples, batch_size, input_size, 1]
        ZK = ZK_and_dW[0]

        #----------------------------------------------------------------------#
        #                      Processing 3rd decode layer
        #----------------------------------------------------------------------#
        if self.num_layer == 3:
            # noises
            dW_0T = ZK_and_dW[1]  # the second one is for the Decode layer 0
            dW_1T = ZK_and_dW[2]  # the third one is for the Decode layer 1
            # dW_2T = ZK_and_dW[3]  # the fourth one is for the Decode layer 2

            # Deconstruct Z^(K) into A^(K), N^(K), Sigma^(K)
            A3_T = ZK[:,:, :, :tf.square(self.size_N_3)]
            A3_T_mat = tf.reshape(A3_T, [tf.shape(A3_T)[0], tf.shape(A3_T)[1], tf.shape(A3_T)[2],
                                         self.size_N_3, self.size_N_3]) # To matrix

            N3_T = ZK[:,:, :, tf.square(self.size_N_3):tf.square(self.size_N_3) + self.size_N_3]

            Sigma_3T = ZK[:,:, :, tf.square(self.size_N_3) + self.size_N_3:]
            Sigma_3T_mat = tf.reshape(Sigma_3T, [tf.shape(Sigma_3T)[0], tf.shape(Sigma_3T)[1], tf.shape(Sigma_3T)[2],
                                                 self.size_N_3, self.size_N_3])  # To matrix

            # [num_samples, input_size_3, 1] + [num_samples, input_size_3,input_size_3] x [num_samples, input_size_3, 1]
            # + [num_samples, input_size_3, 1] + [num_samples, input_size_3,input_size_3] x [num_samples, input_size_3, 1]
            # Z2_T1 = Z2_T + tf.matmul(A3_T_mat, Z2_T) + N3_T + tf.matmul(Sigma_3T_mat, dW_2T)
            Z2_T1 = Z2_T + tf.matmul(A3_T_mat, Z2_T) + N3_T
            # Z2_T1 = self.scaled_tanh(Z2_T1, self.m2)
            A2_T = Z2_T[:,:, :, :tf.square(self.size_N_2)]
            N2_T = Z2_T[:,:, :, tf.square(self.size_N_2):tf.square(self.size_N_2) + self.size_N_2]
            Sigma_2T = Z2_T[:,:, :, tf.square(self.size_N_2) + self.size_N_2:]

        #----------------------------------------------------------------------#
        #                      Processing 2nd decode layer
        #----------------------------------------------------------------------#
        if self.num_layer == 2:
            # noises
            dW_0T = ZK_and_dW[1]  # the second one is for the Decode layer 0
            dW_1T = ZK_and_dW[2]  # the third one is for the Decode layer 1

            A2_T = ZK[:,:, :, :tf.square(self.size_N_2)]
            N2_T = ZK[:,:, :, tf.square(self.size_N_2):tf.square(self.size_N_2) + self.size_N_2]
            Sigma_2T = ZK[:,:, :, tf.square(self.size_N_2) + self.size_N_2:]

        if self.num_layer == 3 or self.num_layer == 2:
            A2_T_mat = tf.reshape(A2_T, [tf.shape(A2_T)[0], tf.shape(A2_T)[1], tf.shape(A2_T)[2],
                                         self.size_N_2, self.size_N_2])  # To matrix

            Sigma_2T_mat = tf.reshape(Sigma_2T, [tf.shape(Sigma_2T)[0], tf.shape(Sigma_2T)[1], tf.shape(Sigma_2T)[2],
                                                 self.size_N_2, self.size_N_2])  # To matrix

            # [input_size_2, 1] + [input_size_2,input_size_2] x [input_size_2, 1] + [input_size_2, 1]
            # + [input_size_2,input_size_2] x [input_size_2, 1]
            # Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T + tf.matmul(Sigma_2T_mat, dW_1T)
            # Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T + tf.matmul(Sigma_2T_mat, dW_1T)
            if self.add_noise_to2nd:
                Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T + tf.matmul(Sigma_2T_mat, dW_1T)
            else:
                Z1_T1 = Z1_T + tf.matmul(A2_T_mat, Z1_T) + N2_T
            # Z1_T1 = self.scaled_tanh(Z1_T1, self.m1)

            A1_T = Z1_T[:,:, :, :tf.square(self.size_N_1)]
            N1_T = Z1_T[:,:, :, tf.square(self.size_N_1):tf.square(self.size_N_1) + self.size_N_1]
            Sigma_1T = Z1_T[:,:, :, tf.square(self.size_N_1) + self.size_N_1:]

        # ---------------------------------------------------------------------- #
        #                      Processing 1st decode layer
        # ---------------------------------------------------------------------- #
        if self.num_layer == 1:
            # noises
            dW_0T = ZK_and_dW[1]  # the second one is for the Decode layer 0

            A1_T = ZK[:, :, :, :tf.square(self.size_N_1)]
            N1_T = ZK[:, :, :, tf.square(self.size_N_1):tf.square(self.size_N_1) + self.size_N_1]
            Sigma_1T = ZK[:, :, :, tf.square(self.size_N_1) + self.size_N_1:]

        if self.num_layer == 3 or self.num_layer == 2 or self.num_layer == 1:
            # [input_size_1, 1] + [input_size_1,input_size_1] x [input_size_1, 1] + [input_size_1, 1]
            # + [input_size_1,input_size_1] x [input_size_1, 1]
            Z0_T1 = Z0_T + tf.matmul(A1_T, Z0_T) + N1_T + tf.matmul(Sigma_1T, dW_0T)
            # Z0_T1 = 1. * tf.tanh((Z0_T + tf.matmul(A1_T, Z0_T) + N1_T) /1.)
            # Z0_T1 = self.scaled_tanh(Z0_T1, self.m0)
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

        if self.num_layer == 1:
            return Z0_T1
        elif self.num_layer == 2:
            return [Z0_T1, Z1_T1]
        elif self.num_layer == 3:
            # T1 means time T+1
            return [Z0_T1, Z1_T1, Z2_T1]
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

    def get_encode_states(self):
        states_over_time = tf.scan(self.Encode, self.scan_inputs, initializer=self.Encoder_initial_state, name='Encode_states')
        return states_over_time

    def get_decode_layerwise_states(self, Decoder_initial_state, ZK_and_dW):
        states_over_time = tf.scan(self.Decode_layerwise, ZK_and_dW, initializer=Decoder_initial_state, name='Decode_states')
        return states_over_time

    def get_decode_states_AvgLoss(self, Decoder_initial_state, ZK_and_dW):
        states_over_time = tf.scan(self.Decode_layerwise_AvgLoss, ZK_and_dW, initializer=Decoder_initial_state,
                                   name='Decode_states')
        return states_over_time

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


def initialize_eta(batch_size, input_size, eta=[0.01, 0.01, 0.01, 0.8, 0.8]):
    """
    initialize the eta, expand them to the same size of batches to ease the element-wise multiplication

    """
    eta_A_t = np.tile(np.expand_dims(np.expand_dims(np.array([eta[0]]), axis=-1), axis=-1),
                       [batch_size, input_size, input_size])
    eta_N_t = np.tile(np.expand_dims(np.expand_dims(np.array([eta[1]]), axis=-1), axis=-1),
                       [batch_size, input_size, 1])
    eta_Sigma_t = np.tile(np.expand_dims(np.expand_dims(np.array([eta[2]]), axis=-1), axis=-1),
                           [batch_size, input_size, input_size])
    phi_t = np.tile(np.expand_dims(np.expand_dims(np.array([eta[3]]), axis=-1), axis=-1),
                     [batch_size, input_size, 1])
    rho_t = np.tile(np.expand_dims(np.expand_dims(np.array([eta[4]]), axis=-1), axis=-1),
                     [batch_size, input_size, input_size])

    return eta_A_t, eta_N_t, eta_Sigma_t, phi_t, rho_t

def generate_noise(num_layer=None, dt=None, RandomState=None, expand_steps=None, num_sub_series=None, sample_times=None,
                   batch_size=None, input_size_1=None, input_size_2=None, is_train=True):
    noise=[]
    # generate noises
    if is_train:
        if num_layer == 1 or num_layer == 2 or num_layer == 3:
            dW_0 = brownian_motion_matrix(dt, RandomState,
                                          size_=[expand_steps, num_sub_series,
                                                 sample_times, batch_size, input_size_1, 1])
            noise.append(dW_0)

        if num_layer == 2 or num_layer == 3:
            dW_1 = brownian_motion_matrix(dt, RandomState,
                                          size_=[expand_steps, num_sub_series,
                                                 sample_times, batch_size, input_size_2, 1])
            noise.append(dW_1)
        # if num_layer == 3:
        #     dW_2 = brownian_motion_matrix(dt, RandomState, expand_steps=expand_steps,
        #                                   num_noise=sample_times, batch_size=batch_size,
        #                                   input_size=input_size_3)
    else:
        if num_layer == 1 or num_layer == 2 or num_layer == 3:
            dW_0 = brownian_motion_matrix(dt, RandomState,
                                          size_=[expand_steps, sample_times, batch_size, input_size_1, 1])
            noise.append(dW_0)
        if num_layer == 2 or num_layer == 3:
            dW_1 = brownian_motion_matrix(dt, RandomState,
                                          size_=[expand_steps, sample_times, batch_size, input_size_2, 1])
            noise.append(dW_1)
        # if num_layer == 3:
        #     dW_2 = brownian_motion_matrix(dt, RandomState, expand_steps=expand_steps,
        #                                   num_noise=sample_times, batch_size=batch_size,
        #                                   input_size=input_size_3)
    return noise

def run_sess_with_graph(cell=None, BayesianOpt=None, acq=None, label=None,
                        batch_size=None, n_samples=None, sample_times=None,
                        loss_mean=None, loss_var=None, loss=None,
                        pred_y=None, input_size_1=None, input_size_2=None, input_size_3=None,
                        graph=None, BayesianOpt_iters=100, scipy_opt='L-BFGS-B',
                        num_warmup=100, num_iters_ForAcqOpt=300, num_layer=None,
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
                        x_=None, y_=None, dw0=None, dw1=None, is_train=True,
                        weighted_param=None, weights=None):
    # if in test model, we don't iteratively find the best x_next
    if not is_train:
        BayesianOpt_iters = 1

    for iter in range(1, BayesianOpt_iters + 1):
        feed_ = {}

        if is_train:
            x_next = BayesianOpt.find_next_x(acq, opt_method=scipy_opt,
                                             n_warmup=num_warmup, n_iters=num_iters_ForAcqOpt
                                             )
        if num_layer == 1:

            if is_train:
                A1_0, N1_0, Sigma_10, eta_A1_t, eta_N1_t, eta_Sigma_1t, phi_1t, rho_1t \
                    = rebuild_chosen_initial_state_layerwise(x_next, batch_size, input_size_1, layer=1, if_ANSigma=True)
                Z1_0 = np.concatenate((A1_0, N1_0, Sigma_10), axis=1)
                feed_.update({weighted_param: weights})

            feed_.update( {
                # weight: weight_value,
                n_samples: sample_times,
                cell.Z0_0: Z0_0,
                cell.ori_inputs: x_,
                label: y_,
                cell.Z1_0: Z1_0,
                cell.A1_0: A1_0,
                cell.N1_0: N1_0,
                cell.Sigma_10: Sigma_10,
                cell.E_epsilon_10: E_epsilon_10,
                cell.cov_epsilon_10: cov_epsilon_10,
                cell.dW_0: dw0,

                cell.eta_A1_t: eta_A1_t,
                cell.eta_N1_t: eta_N1_t,
                cell.eta_Sigma_1t: eta_Sigma_1t,
                cell.phi_1t: phi_1t,
                cell.rho_1t: rho_1t
            } )

        elif num_layer == 2:

            if is_train:
                A2_0, N2_0, Sigma_20, eta_A2_t, eta_N2_t, eta_Sigma_2t, phi_2t, rho_2t \
                    = rebuild_chosen_initial_state_layerwise(x_next, batch_size, input_size_2, layer=2, if_ANSigma=True)
                Z2_0 = np.expand_dims(
                    np.expand_dims(np.concatenate((A2_0.flatten(), N2_0.flatten(), Sigma_20.flatten())), axis=0), axis=-1)
                feed_.update({weighted_param: weights})

            feed_.update( {
                # weight: weight_value,
                n_samples: sample_times,
                cell.Z0_0: Z0_0,
                cell.ori_inputs: x_,
                label: y_,
                cell.Z1_0: Z1_0,
                cell.A1_0: A1_0,
                cell.N1_0: N1_0,
                cell.Sigma_10: Sigma_10,
                cell.E_epsilon_10: E_epsilon_10,
                cell.cov_epsilon_10: cov_epsilon_10,
                cell.dW_0: dw0,

                cell.Z2_0: Z2_0,
                cell.A2_0: A2_0,
                cell.N2_0: N2_0,
                cell.Sigma_20: Sigma_20,
                cell.E_epsilon_20: E_epsilon_20,
                cell.cov_epsilon_20: cov_epsilon_20,
                cell.dW_1: dw1,

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
            } )

        elif num_layer == 3:

            if is_train:

                eta_A3_t, eta_N3_t, eta_Sigma_3t, phi_3t, rho_3t \
                    = rebuild_chosen_initial_state_layerwise(x_next, batch_size, input_size_3, layer=3, if_ANSigma=False)

                Z3_0 = np.expand_dims(
                    np.expand_dims(np.concatenate((A3_0.flatten(), N3_0.flatten(), Sigma_30.flatten())), axis=0),
                    axis=-1)
                feed_.update({weighted_param: weights})

            feed_.update( {
                # weight: weight_value,
                n_samples: sample_times,
                cell.Z0_0: Z0_0,
                cell.ori_inputs: x_,
                label: y_,
                cell.Z1_0: Z1_0,
                cell.A1_0: A1_0,
                cell.N1_0: N1_0,
                cell.Sigma_10: Sigma_10,
                cell.E_epsilon_10: E_epsilon_10,
                cell.cov_epsilon_10: cov_epsilon_10,
                cell.dW_0: dw0,

                cell.Z2_0: Z2_0,
                cell.A2_0: A2_0,
                cell.N2_0: N2_0,
                cell.Sigma_20: Sigma_20,
                cell.E_epsilon_20: E_epsilon_20,
                cell.cov_epsilon_20: cov_epsilon_20,
                cell.dW_1: dw1,

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
            } )
        else:
            raise NotImplementedError("Does not support num_layer which is not equal to 1, 2 or 3")

        # sess.close()

        tf.reset_default_graph()

        with tf.Session(graph=graph) as sess:

            if is_train:

                output_loss_mean, output_loss_var, output_loss_WithSigma \
                    = sess.run([loss_mean, loss_var, loss],
                               feed_dict=feed_
                               )
                yield x_next, output_loss_mean, output_loss_var, output_loss_WithSigma
            else:

                output_loss_mean, output_loss_var, output_loss_WithSigma, output_pred_y_ForTrain \
                    = sess.run([loss_mean, loss_var, loss, pred_y],
                               feed_dict=feed_
                               )
                yield output_loss_mean, output_loss_var, output_loss_WithSigma, output_pred_y_ForTrain