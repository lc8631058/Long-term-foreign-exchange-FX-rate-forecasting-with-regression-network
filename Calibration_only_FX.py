import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import cholesky
from sklearn.linear_model import LinearRegression
# from Calibration import Find_max_min_from_parameters
# from Generate_artificial_data import transform_X_to_dataframe
# from artificial_data_third_edition import generate_X_GOU_2, generate_X_GOU_2_with_set_params

def calibration_fx(data, dt, if_log=True):
    """
    calibrate the a and b parameters of equation:
    y = a*x_{1} + b*x_{2} + c + \epsilon
    data: The Dataframe
    x2: x2 is the second feature
    if_log: If the data must be non-negative: True, else False
    :return: alpha, m, epsilon
    """

    # x1 is t
    # data_len = len(data['EURUSD_Curncy'])
    # time_ = np.arange(1.0, data_len).reshape(-1, 1) # the first feature is time

    y_t = np.array([])

    for i, key_ in enumerate(data):
        if if_log:
            data_i_ = np.log(np.array(np.copy(data[key_]))).reshape((-1, 1))
        else:
            data_i_ = np.array(np.copy(data[key_])).reshape((-1, 1))
        y_t = np.concatenate((y_t, data_i_), axis=1) if y_t.size else data_i_ #concatenate

    # y for LinearRegression
    label_ = np.array(y_t[1:, :] - y_t[:-1, :])
    # label_ = np.concatenate((time_, label_), axis=1)
    # x for LinearRegression
    # train_ = y_t[:-1, :].reshape(-1,1)
    train_ = y_t[:-1, :]
    # train_ = np.concatenate((time_, train_), axis=1)
    # if calibration_target in if_Non_Nega and if_Non_Nega[calibration_target]:
    #     x = np.log(np.array(np.copy(data[calibration_target][:-1]))).reshape(-1,1)
    # else:
    #     x = np.array(np.copy(data[calibration_target][:-1])).reshape(-1, 1)

    # LinearRegression fit
    reg = LinearRegression().fit(train_, label_)
    # predict
    y_pred = reg.predict(train_) # predict y using x
    epsilon = (label_ - y_pred)
    # calculate R2 score
    R2_score = reg.score(train_, label_)

    # X2 = sm.add_constant(train_)
    # est = sm.OLS(label_[:,0], X2)
    # est2 = est.fit()
    # print(est2.summary())
    # print(est2.pvalues)

    # y = ax + b + epsilon
    # M = reg.coef_[:, 1] / dt
    # A = reg.coef_[:, 1:] / dt

    A = reg.coef_ / dt
    N = reg.intercept_ / dt

    if epsilon.shape[-1] == 1:
        L = np.std(epsilon)
        L = np.expand_dims(np.expand_dims(L, axis=-1), axis=-1)
    else:
        # Covariance Matrix of Epsilon
        cov_matrix = np.cov(np.transpose(epsilon))
        L = cholesky(cov_matrix / dt)

    # return M, A, N, L, R2_score
    return A, N, L, R2_score


def multi_calibration_fx(full_data, dt, interval_idx, num_years):
    """
    Calibrate on moving windows with size num_years, move interval is one year
    :param full_data: full saved data
    :param dt: delta t
    :param Non_Nega: A dict, key is financial factors, value is if it must be Non-nagative values
    :param interval_idx: saved yearly interval index
    :param num_years: number of years to calibrate
    :return: parameters calculated from each moving window,
             e.g. Alpha.shape[0] is the number of possible moving windows over data
                  Alpha.shape[1] is the number of financial factors
    """
    Alpha_ = []
    # M_ = []
    N_ = []
    # Epsilon_ = []
    # cov_matrix_ = []
    L_ = []
    len_ = len(interval_idx)
    # interval_idx -=
    for i in range(len_ - num_years):
        # create clipped data
        clipped_data = {}
        for l, key in enumerate(full_data):
            clipped_data[key] = full_data[key][interval_idx[i]: interval_idx[num_years+i]]

        # M, A, N, L, R2_score = calibration_fx(clipped_data, dt)
        A, N, L, R2_score = calibration_fx(clipped_data, dt)
        Alpha_.append(A)
        # M_.append(M)
        N_.append(N)
        # Epsilon_.append(Epsilon)
        # cov_matrix_.append(cov_matrix)
        L_.append(L)
        # Note here each moving window has different size (cause each year has different number of days)
        # so I will not transform it to np.array
    # return np.array(M_), np.array(Alpha_), np.array(N_), np.array(L_)
    return np.array(Alpha_), np.array(N_), np.array(L_)


#---------------------------------------------------------------------------------------#
#               Calibrate only one currency serie for test OnlineLR algorithms
#---------------------------------------------------------------------------------------#
dt = 1.0
test_for_OnlineLR_1 = True
# test_for_OnlineLR_1 = False
if test_for_OnlineLR_1:
    # load full data
    with open("./Data/calibration parameters/data_FX_dayly.pkl", "rb") as f:
        data = pickle.load(f)

    # data = pd.read_csv("./Data/SP500/SP500_GSPC_daily.csv", sep=';')

    cny_ = {}
    A_list = []
    N_list = []
    L_list = []
    for i_ in range(0, 4975, 100):
        cny_['EURCNY_Curncy'] = data['EURCNY_Curncy'][i_:500+i_]
        # cny_['Close'] = data['Close'][i_:200 + i_]

        A, N, L, R2_score = calibration_fx(cny_, dt, if_log=True)
        A_list.append(A)
        N_list.append(N)
        L_list.append(L)

    x_t = np.array(np.copy(data['EURCNY_Curncy'][:500])).reshape((-1, 1))

    # save serie and calibrated label
    with open("./Data/artificial_fx_2/for_training_OnlineLR/single_serie_for_OnlineLR_CNY_1Y.pkl", "wb") as f:
        pickle.dump((x_t, A, N, L), f)

#---------------------------------------------------------------------------------------#
#               Calibrate time series for test OnlineLR algorithms
#---------------------------------------------------------------------------------------#
#
# test_for_OnlineLR = True
test_for_OnlineLR = False
if test_for_OnlineLR:

    # load full data
    with open("./Data/calibration parameters/data_FX_dayly.pkl","rb") as f:
        data = pickle.load(f)

    # DataFrame
    clipped_data = {}
    for l, key in enumerate(data):
        clipped_data[key] = data[key][1000:1000+200]

    A, N, L, R2_score = calibration_fx(clipped_data, dt, if_log=True)

    # transform DataFrame to ndarray
    x_t = np.array([])

    for i, key_ in enumerate(clipped_data):
        data_i_ = np.array(np.copy(clipped_data[key_])).reshape((-1, 1))
        x_t = np.concatenate((x_t, data_i_), axis=1) if x_t.size else data_i_ #concatenate

    delta_x = np.log(x_t[1:]) - np.log(x_t[:-1])
    # save serie and calibrated label
    with open("./Data/artificial_fx_2/for_training_OnlineLR/real_data_for_OnlineLR.pkl", "wb") as f:
        pickle.dump((x_t, A, N, L), f)

    # plot for comparison
    with open("./Data/artificial_fx_2/for_training_OnlineLR/arti_0", "rb") as f:
        arti_ = pickle.load(f)

    t_ = np.linspace(0, 1829, num=1829)
    plt.figure()
    plt.plot(t_, delta_x, label='delta_y_t')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(t_, np.log(x_t[:-1]), label='y_t_1')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.xlabel('y_t_1')
    plt.ylabel('delta_y_t')
    plt.scatter(np.log(x_t[:-1]), delta_x, label='delta_y_t vs y_t_1')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(t_, np.log(arti_[0, 1:]) - np.log(arti_[0, :-1]), label='synthetic delta_y_t')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(t_, np.log(arti_[0, :-1]), label='synthetic y_t_1')
    plt.grid()
    plt.legend()
    plt.figure()
    plt.xlabel('synthetic y_t_1')
    plt.ylabel('synthetic delta_y_t')
    plt.scatter(np.log(arti_[0, 1:]) - np.log(arti_[0, :-1]), np.log(arti_[0, :-1]), label='synthetic delta_y_t vs y_t_1')
    plt.legend()
    plt.grid()
    plt.show()
#=======================================================================================#
#                        Multi-calibration for generalized OU
#=======================================================================================#
#
# if_multi_calibration_for_GOU = True
if_multi_calibration_for_GOU = False
if if_multi_calibration_for_GOU:

    # load full data
    f = open("./Data/calibration parameters/data_FX_dayly.pkl","rb") # writting and binary
    data = pickle.load(f)
    f.close()

    # # load interval_idx
    # f = open("./Data/calibration parameters/interval_idx.pkl","rb") # writting and binary
    # interval_idx = pickle.load(f)
    # f.close()
    interval_idx = [0, 269, 516, 777, 1042, 1304, 1564, 1824, 2085, 2363, 2605, 2866, 3126, 3387, 3648, 3909, 4170, 4431, 4691, 4952]
    for num_year in range(1,19): # The number of years
        # M, A, N, L = multi_calibration_fx(data, dt, interval_idx, num_year)
        A, N, L = multi_calibration_fx(data, dt, interval_idx, num_year)
        # save the calibration result
        with open("./Data/calibration parameters/multi_calibration_GOU_FX/calib_parameters_GOU_{}y.pkl"\
                   .format(num_year),"wb") as f:
            # pickle.dump({'M':M, 'Alpha':A, 'N':N, 'L':L}, f)
            pickle.dump({'Alpha': A, 'N': N, 'L': L}, f)


#=======================================================================================#
#                    Find max_final and min_final
#=======================================================================================#

# find_reject_outliers = True
find_reject_outliers = False
if find_reject_outliers:
    key = ['Alpha', 'N', 'L']
    path = "./Data/calibration parameters/multi_calibration_GOU_FX/calib_parameters_GOU_{}y.pkl"
    # num_years = [5,6,7,8,9,10]
    max_final, min_final = Find_max_min_from_parameters(file_path=path, key_=key, num_year=[1, 19], bias=0.0)

    with open("./Data/calibration parameters/multi_calibration_GOU_FX/parameters range/max_parameters.pkl", "wb") as f: # writting and binary
        pickle.dump(max_final, f)
    with open("./Data/calibration parameters/multi_calibration_GOU_FX/parameters range/min_parameters.pkl", "wb") as f: # writting and binary
        pickle.dump(min_final, f)



#=======================================================================================#
#                         Generate data
#=======================================================================================#
#
# draw_out = True
draw_out = False
# Generate_outliers = True
Generate_outliers = False
# random_seed = 1 # important, make sure to generate deterministic random process
if Generate_outliers:
    with open("./Data/calibration parameters/multi_calibration_GOU_FX/parameters range/max_parameters.pkl", "rb") as f: # writting and binary
        max_final = pickle.load(f)
    with open("./Data/calibration parameters/multi_calibration_GOU_FX/parameters range/min_parameters.pkl", "rb") as f: # writting and binary
        min_final = pickle.load(f)
    #max_final['N']
    # with open("./Data/calibration parameters/multi_calibration_generalized_OU/parameters range/max_parameters_1_0.pkl", "rb") as f: # writting and binary
    #     max_final = pickle.load(f)
    # with open("./Data/calibration parameters/multi_calibration_generalized_OU/parameters range/min_parameters_1_0.pkl", "rb") as f: # writting and binary
    #     min_final = pickle.load(f)

    # with open("./Data/calibration parameters/multi_calibration_generalized_OU/parameters range/max_parameters_1_5.pkl", "rb") as f: # writting and binary
    #     max_final = pickle.load(f)
    # with open("./Data/calibration parameters/multi_calibration_generalized_OU/parameters range/min_parameters_1_5.pkl", "rb") as f: # writting and binary
    #     min_final = pickle.load(f)

    # load data
    with open("./Data/calibration parameters/data_FX_dayly.pkl", "rb") as f:  # writting and binary
        data = pickle.load(f)
    stretching = 2.
    factors_range = { 'EURUSD_Curncy': [0.8255 / stretching, 1.599 * stretching], 'EURGBP_Curncy': [0.5736 / stretching, 0.9793 * stretching],
                      'EURCNY_Curncy': [6.5583 / stretching, 11.2861 * stretching]
                      }

    # factors_range = {'EURUSD_Curncy': [0.8255 / 1.5, 1.599 * 1.5], 'EURGBP_Curncy': [0.5736 / 1.5, 0.9793 * 1.5],
    #                  'EURCNY_Curncy': [6.5583 / 1.5, 11.2861 * 1.5]
    #                  }


    # mxa = max_final['Alpha']
    # mia = min_final['Alpha']
    # mxm = np.concatenate((np.exp(max_final['M'][:6]), max_final['M'][6:]))
    # mim = np.concatenate((np.exp(min_final['M'][:6]), min_final['M'][6:]))
    # mxl = max_final['L']
    # mil = min_final['L']
    K = 3  # Number of brownian motion
    dt = 1.0
    N = 366 * 5 - 1 # one position for y0
    len_data = len(data['EURUSD_Curncy']) # used for generate random y0

    if_Non_Nega = {'EURUSD_Curncy':True, 'EURGBP_Curncy':True, 'EURCNY_Curncy':True}

    if_abonden = False

    # tried_ = 0
    # np.random.seed(random_seed)
    # Nan_ = np.float64('NaN')
    # Inf_ = np.float64('Inf')
    for n_files in range(100):  # separately save data
        arti_data_ = []
        parameters_ = []
        num_qualified_serie = 0

        while num_qualified_serie != 200: # generate 1000 data
        # for i in range(1):
            if_abonden = False
            paras, X = generate_X_GOU_2(data, len_data, if_Non_Nega, max_final, min_final, K, dt, N, if_log=True)

            # judge if generated serie is in compliance with the reality, e.g. no infinite number
            for idx_, key_ in enumerate(factors_range):
                lower_bound = factors_range[key_][0]
                upper_bound = factors_range[key_][1]
                min_X = abs(np.amin(X[idx_, :]))
                max_X = np.amax(X[idx_, :])
                # w = np.isnan(min_X)
                # ww = np.float64('NaN')

                # alpha_ = paras[0][idx_]
                # n_ = paras[2][idx_]
                # sigma = paras[3][idx_]

                if min_X < lower_bound or max_X > upper_bound or np.isnan(min_X) or np.isnan(max_X)\
                   or np.isinf(min_X) or np.isinf(max_X):
                    if_abonden = True
                    # tried_ += 1
                    # print("The {} times try failed".format(tried_))
                    break

            if if_abonden:
                continue

            # transform generated data to DataFrame file
            column = np.array(['EURUSD_Curncy', 'EURGBP_Curncy', 'EURCNY_Curncy'])
            X = transform_X_to_dataframe(X, column)
            parameters_.append(paras)
            arti_data_.append(X)

            num_qualified_serie += 1
            print("file {} / time serie: {} added".format(n_files, num_qualified_serie))


            # Calculate the parameter to check if the model is right
            # alpha = paras[0]
            # n = paras[1]
            # m = np.matmul(-1. * np.linalg.inv(paras[0]), paras[1])
            # ay_n = np.matmul(alpha, m) + n

            if draw_out:
                t5 = X['EURUSD_Curncy'].index.values
                for i, legd in enumerate(if_Non_Nega):
                    plt.figure(legd)
                    plt.title(legd)
                    # if if_Non_Nega[legd]:
                    #     M = np.exp(paras[1][i])
                    # else:
                    #     M = paras[1][i]
                    plt.plot(t5, X[legd], label=r'\$\alpha$: USD {:5.2E},{:5.2E},{:5.2E} \
                                                    $\alpha: $GBP {:5.2E},{:5.2E},{:5.2E} \
                                                    $\alpha$: CNY {:5.2E},{:5.2E},{:5.2E} \
                                                    n: USD {:5.2E}, GBP {:5.2E}, CNY {:5.2E} \
                                                    m: USD {:5.2E}, GBP {:5.2E}, CNY {:5.2E} \
                                                    ay_n: {:5.2E}, {:5.2E}, {:5.2E}'
                             .format(alpha[0][0], alpha[0][1], alpha[0][2],
                                     alpha[1][0], alpha[1][1], alpha[1][2],
                                     alpha[2][0], alpha[2][1], alpha[2][2],
                                     n[0], n[1], n[2],
                                     m[0], m[1], m[2],
                                     ay_n[0], ay_n[1], ay_n[2]))
                    plt.grid(True)
                    plt.legend()

                plt.show()

        with open("./Data/artificial_fx_2/original/arti_{}".format(n_files), "wb") as f:
            pickle.dump(arti_data_, f)
        with open("./Data/artificial_fx_2/original/paras_{}".format(n_files), "wb") as g:
            pickle.dump(parameters_, g)



        # t1 = np.linspace(1, len(paras[0]), num=len(paras[0]))
        # plt.figure('Alpha')
        # plt.plot(t1, paras[0], label='Alpha')
        # plt.title('Alpha')
        # plt.xlabel('n', fontsize=16)
        # plt.ylabel('Alpha', fontsize=16)
        # plt.legend()
        # plt.grid(True)
        #
        # t2 = np.linspace(1, len(paras[1]), num=len(paras[1]))
        # plt.figure('M')
        # plt.plot(t2, paras[1], label='M')
        # plt.title('M')
        # plt.xlabel('n', fontsize=16)
        # plt.ylabel('M', fontsize=16)
        # plt.legend()
        # plt.grid(True)
        #
        # t3 = np.linspace(1, len(paras[2]), num=len(paras[2]))
        # plt.figure('N')
        # plt.plot(t3, paras[2], label='N')
        # plt.title('N')
        # plt.xlabel('n', fontsize=16)
        # plt.ylabel('N', fontsize=16)
        # plt.legend()
        # plt.grid(True)
        #
        # t4 = np.linspace(1, len(extract_lower_triangular(paras[3])), num=len(extract_lower_triangular(paras[3])))
        # plt.figure('Sigma')
        # plt.plot(t4, extract_lower_triangular(paras[3]), label='Sigma')
        # plt.title('Sigma')
        # plt.xlabel('n', fontsize=16)
        # plt.ylabel('Sigma', fontsize=16)
        # plt.legend()
        # plt.grid(True)







#=======================================================================================#
#                         Evaluate difference of parameters
#=======================================================================================#
#

def evaluate_delta_for_parameter_generating(param_):

    Alpha_ = param_['Alpha']
    Alpha_.sort(axis=0)
    delta_Alpha = np.mean(Alpha_[1:, :, :] - Alpha_[:-1, :, :], axis=0) # mean increment when generating parameters

    N_ = param_['N']
    N_.sort(axis=0)
    delta_N = np.mean(N_[1:, :] - N_[:-1, :], axis=0)

    Sigma_ = param_['L']
    Sigma_.sort(axis=0)
    delta_Sigma = np.mean(Sigma_[1:, :, :] - Sigma_[:-1, :, :], axis=0)

    return delta_Alpha, delta_N, delta_Sigma


def generate_parameters(param_, delta_Alpha, delta_N, delta_Sigma, initial=9):
    # Alpha
    arti_Alpha = np.zeros(shape=[10, param_['Alpha'].shape[1], param_['Alpha'].shape[2]])

    real_Alpha = param_['Alpha'][initial]
    for i in range(5):
        real_Alpha -= delta_Alpha
        arti_Alpha[i, :, :] = real_Alpha

    real_Alpha = param_['Alpha'][initial]
    for i in range(5,10):
        real_Alpha += delta_Alpha
        arti_Alpha[i, :, :] = real_Alpha

    # N
    arti_N = np.zeros(shape=[10, param_['N'].shape[1]])

    real_N = param_['N'][initial]
    for i in range(5):
        real_N -= delta_N
        arti_N[i, :] = real_N

    real_N = param_['N'][initial]
    for i in range(5, 10):
        real_N += delta_N
        arti_N[i, :] = real_N

    # Sigma
    arti_Sigma = np.zeros(shape=[10, param_['L'].shape[1], param_['L'].shape[2]])

    real_Sigma = param_['L'][initial]
    for i in range(5):
        real_Sigma -= delta_Sigma
        arti_Sigma[i, :, :] = real_Sigma

    real_Sigma = param_['L'][initial]
    for i in range(5, 10):
        real_Sigma += delta_Sigma
        arti_Sigma[i, :, :] = real_Sigma

    return arti_Alpha, arti_N, arti_Sigma

# with open("./Data/calibration parameters/multi_calibration_GOU_FX/calib_parameters_GOU_1y.pkl","rb") as f:
#     param_ = pickle.load(f)
#
# key_ = ['Alpha', 'N', 'L']

# delta_Alpha, delta_N, delta_Sigma = evaluate_delta_for_parameter_generating(param_)
# arti_Alpha, arti_N, arti_Sigma = generate_parameters(param_, delta_Alpha*2, delta_N*2, delta_Sigma*2)



#=======================================================================================#
#                         Generate data with set parameters
#=======================================================================================#
#
# do_ = True
do_ = False
if do_:
    # with open("./Data/calibration parameters/multi_calibration_GOU_FX/calib_parameters_GOU_1y.pkl","rb") as f:
    #     param_ = pickle.load(f)
    with open("./Data/artificial_fx/paras_0", "rb") as f:
        param_ = pickle.load(f)

    with open("./Data/calibration parameters/data_FX_dayly.pkl", "rb") as f:  # writting and binary
        data = pickle.load(f)
    K = 3  # Number of brownian motion
    dt = 1.0
    N = 366 * 5 - 1 # one position for y0
    len_data = len(data['EURUSD_Curncy']) # used for generate random y0

    if_Non_Nega = {'EURUSD_Curncy':True, 'EURGBP_Curncy':True, 'EURCNY_Curncy':True}

    for which_ in range(10):

        p_s, X_ = generate_X_GOU_2_with_set_params(data, len_data, if_Non_Nega, K, dt, N, alpha, n, sigma, if_log=True)

        column = np.array(['EURUSD_Curncy', 'EURGBP_Curncy', 'EURCNY_Curncy'])
        X_ = transform_X_to_dataframe(X_, column)

        t5 = X_['EURUSD_Curncy'].index.values
        for i, legd in enumerate(if_Non_Nega):
            plt.figure(legd)
            plt.title(legd)
            # if if_Non_Nega[legd]:
            #     M = np.exp(paras[1][i])
            # else:
            #     M = paras[1][i]
            plt.plot(t5, X_[legd], label=r'$\alpha$:{:5.2E},{:5.2E},{:5.2E}; N:{:5.2E}'\
                                        .format(alpha[i][0], alpha[i][1], alpha[i][2], n[i]))
            # plt.plot(t5, X[legd], label=r'artificial data')
            plt.grid(True)
            plt.legend()

        plt.show()

#=======================================================================================#
#                         Evaluate the generated data
#=======================================================================================#
#
eva_ = False
# eva_ = True
if eva_:
    with open("./Data/artificial_fx_2/original/arti_1","rb") as f:
        data_ = pickle.load(f)
    with open("./Data/artificial_fx_2/original/paras_1","rb") as f:
        paras_ = pickle.load(f)

    idx = 3

    A_, N_, Sigma_, R2_score_ = calibration_fx(data_[idx], dt)
    A_label = paras_[idx][0]
    N_label = paras_[idx][1]
    Sigma_label = paras_[idx][2]

    print("A_cali: {}\n\n A_label: {}\n".format(A_, A_label))
    print('\n')
    print("N_cali: {}\n\n N_cali: {}\n".format(N_, N_label))
    print('\n')
    print("Sigma_cali: {}\n\n Sigma_cali: {}\n".format(Sigma_, Sigma_label))

# Verified that the calibration is right