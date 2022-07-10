import os
import time
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
import GPyOpt
from scipy import stats
from RegPred_function import Multi_layer_RegPred, genrate_bounds_GpyOpt, build_graph_ForPred, build_graph_ForTrain
from RegPred_Cell import initialization, generate_noise
from utils import confident_interval_accuracy
from RegPred_function import collector

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # avoid gpu

if_Non_Nega = {'EURCNY_Curncy': True}
num_factors = len(if_Non_Nega) # number of financial factors
serie_steps = 1830 # 366*5 days
# if_Non_Nega = {'EURUSD_Curncy': True, 'EURGBP_Curncy': True, 'EURCNY_Curncy': True}

len_Sigma = num_factors**2 # length of Sigma without zeros
len_Alpha = num_factors**2
len_N = num_factors
len_params = len_Alpha + len_N + len_Sigma # number of parameters to predict

epochs = 1
# LossCalculation_steps = 100
# future_steps = 100
expand_steps_ForTrain = 200 # the steps you wanna decoder to expand
expand_steps_ForPred = 100
num_noises = 3 # number of noises depends on number of Decoder layers minus 1, which is layer 2, layer 1, layer 0

# batch size
batch_size = 1

# input size
input_size_1 = 1
input_size_2 = 3
input_size_3 = 21
input_size_4 = 903
size_N_3 = 21
size_N_2 = 3
size_N_1 = 1
input_dim_1 = 5+size_N_2
input_dim_2 = 5+size_N_3

seed = 1

# load data
kind_ = 'USD'
num_years = 5
path = "./Data/RegPred_dataset/RegPred_{}Y".format(num_years)

dt = 1.0
random_state = 1
RandomState_instance = np.random.RandomState(seed=random_state)

# Build the graph
g_ForTrain_1, num_samples_ForTrain_1, label_y_ForTrain_1, \
    RegPred_Cell_1_ForTrain, pred_y_WithSigma_1_ForTrain, loss_WithSigma_1_ForTrain, loss_mean_1_ForTrain, \
    loss_var_1_ForTrain, encode_states_1_ForTrain, ZK_1_ForTrain, Decoder_initial_state_1_ForTrain, \
    decode_states_WithSigma_1_ForTrain, w_loss_mean_1, w_loss_var_1, w_loss_WithSigma_1, weighted_1\
    = build_graph_ForTrain(expand_steps_ForTrain, num_l=1)

g_ForTrain_2, num_samples_ForTrain_2, label_y_ForTrain_2, \
    RegPred_Cell_2_ForTrain, pred_y_WithSigma_2_ForTrain, loss_WithSigma_2_ForTrain, loss_mean_2_ForTrain, \
    loss_var_2_ForTrain, encode_states_2_ForTrain, ZK_2_ForTrain, Decoder_initial_state_2_ForTrain, \
    decode_states_WithSigma_2_ForTrain, w_loss_mean_2, w_loss_var_2, w_loss_WithSigma_2, weighted_2\
    = build_graph_ForTrain(expand_steps_ForTrain, num_l=2)


g_ForPred_1, num_samples_ForPred_1, label_y_ForPred_1, \
    RegPred_Cell_1_ForPred, pred_y_WithSigma_1_ForPred, loss_WithSigma_1_ForPred, loss_mean_1_ForPred,\
    loss_var_1_ForPred, encode_states_1_ForPred, ZK_1_ForPred, Decoder_initial_state_1_ForPred,\
    decode_states_WithSigma_1_ForPred = build_graph_ForPred(expand_steps_ForPred, num_l=1)

g_ForPred_2, num_samples_ForPred_2, label_y_ForPred_2, \
    RegPred_Cell_2_ForPred, pred_y_WithSigma_2_ForPred, loss_WithSigma_2_ForPred, loss_mean_2_ForPred,\
    loss_var_2_ForPred, encode_states_2_ForPred, ZK_2_ForPred, Decoder_initial_state_2_ForPred,\
    decode_states_WithSigma_2_ForPred = build_graph_ForPred(expand_steps_ForPred, num_l=2)

sigma_scale = 0.1
_, _, _, E_EPSILON_10, COV_EPSILON_10 = initialization(batch_size, input_size_1, sigma_scale)
_, _, _, E_EPSILON_20, COV_EPSILON_20 = initialization(batch_size, input_size_2, sigma_scale)
# A3_0, N3_0, SIGMA_30, E_EPSILON_30, COV_EPSILON_30 = initialization(batch_size, input_size_3, sigma_scale)

var_weight = 1.
decay = 1.
WEIGHTS = [1*(decay**ii) for ii in range(serie_steps-1)]
WEIGHTS = np.reshape(WEIGHTS[::-1], [len(WEIGHTS), 1, 1])

# upper bound for different initial states
# A_bound = [-0.08, 0.03]
# N_bound = [-0.005, 0.03]
# Sigma_bound = [0.001, 0.008]
# lr_bound = [0.0001, 0.03]
# phi_rho_bound = [0.1, 1.]
#
# A_bound_2 = [-0.1, 0.1]
# N_bound_2 = [-0.1, 0.1]
# Sigma_bound_2 = [-0.001, 0.001]
# lr_bound_2 = [0.0001, 0.03]
# phi_rho_bound_2 = [0.1, 1.]
A_bound = [-0.8, 0.8]
N_bound = [-0.8, 0.8]
Sigma_bound = [0.001, 0.01]
lr_bound = [0.0001, 0.8]
phi_rho_bound = [0.5, 1.]

A_bound_2 = [-0.2, 0.2]
N_bound_2 = [-0.2, 0.2]
Sigma_bound_2 = [-0.001, 0.001]
lr_bound_2 = [0.0001, 0.8]
phi_rho_bound_2 = [0.5, 1.]

# number of iterations for Bayesian optimization
BayesianOpt_iters = 100
BayesianOpt_iters_2 = 100

SAMPLE_TIMES_ForTrain = 30
SAMPLE_TIMES_ForPred = 100

bounds_1 = genrate_bounds_GpyOpt(layer_idx=1, size=input_size_1, A_bound=A_bound, N_bound=N_bound,
                                 Sigma_bound=Sigma_bound, lr_bound=lr_bound, phi_rho_bound=phi_rho_bound)

bounds_2 = genrate_bounds_GpyOpt(layer_idx=2, size=input_size_2, A_bound=A_bound_2, N_bound=N_bound_2,
                                 Sigma_bound=Sigma_bound_2, lr_bound=lr_bound_2, phi_rho_bound=phi_rho_bound_2)
prior = 'GP' # 'sparseGP' #
acq_func = 'EI' # 'MPI' #' # 'LCB'
exact_f = True # whether the outputs are exact.
acq_optimizer = 'lbfgs' # 'DIRECT'
# kernel = GPy.kern.Matern32(1)
# kernel = k1 = GPy.kern.RBF(1)
kernel = None

# np.random.seed(1)

optimized_list_1, optimized_list_2 = [], []
# CNY: [10, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 36, 38, 39, 42, 43, 47,
#       58, 59, 60, 62, 64, 66, 67, 68, 69, 70, 71, 78, 82, 83, 85, 86, 89, 93]
# USD: [4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36,
#       37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 73, 78,
#       79, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92]
# for ii in range(4, 95+1):
for ii in [8, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 73, 78,
      79, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 4, 5, 6, 7, ]:
    with open(path + '/RegPred_TrainX_{}_{}Y_{}days_AvgLoss.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        train_x_dataset = pkl.load(f)
    with open(path + '/RegPred_TrainY_{}_{}Y_{}days_AvgLoss.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        train_y_ForLossCalculation_dataset = pkl.load(f)
    with open(path + '/RegPred_TestY_{}_{}Y_{}days_AvgLoss.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        train_y_dataset = pkl.load(f)

    Z0_0 = np.expand_dims(np.expand_dims(train_x_dataset[ii - 1, 0], axis=-1), axis=0)  # Z0_0 = Y0

    TRAIN_X = np.expand_dims(train_x_dataset[ii - 1], axis=0)  # add one dimension as the wrapper
    TRAIN_Y_ForLossCalculation = np.expand_dims(train_y_ForLossCalculation_dataset[ii - 1],
                                                axis=-1)  # add one dimension as the wrapper
    # test x is same as train_y for loss calculation
    TRAIN_X_ForPred = np.expand_dims(np.log(train_y_ForLossCalculation_dataset[ii - 1]), axis=0)
    TRAIN_Y = np.expand_dims(train_y_dataset[ii - 1], axis=-1)  # add one dimension as the wrapper

    del train_x_dataset, train_y_ForLossCalculation_dataset, train_y_dataset

    t_expand_steps = np.linspace((serie_steps - 1 + expand_steps_ForTrain) - 1,
                                 serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1,
                                 expand_steps_ForPred + 1)
    t_ = np.linspace(0, TRAIN_X_ForPred.shape[1] + TRAIN_Y.shape[0] - 1,
                     TRAIN_X_ForPred.shape[1] + TRAIN_Y.shape[0])

    # ------------------------------------------------#
    #              Train the 1st layer
    # ------------------------------------------------#
    # train_1st=False
    train_1st = True

    if train_1st:
        start_1 = time.time()
        print('time count begin for {}th, 1st layer training'.format(ii))

        noise_ForTrain = generate_noise(num_layer=1, dt=dt, RandomState=RandomState_instance,
                                        expand_steps=expand_steps_ForTrain, num_sub_series=serie_steps - 1,
                                        sample_times=SAMPLE_TIMES_ForTrain, batch_size=batch_size,
                                        input_size_1=input_size_1, is_train=True)

        collector_1 = collector()
        RegPred_ForTrain_1 = Multi_layer_RegPred(1, RegPred_Cell_1_ForTrain, g_ForTrain_1, loss_mean_1_ForTrain,
                                               loss_var_1_ForTrain, num_samples_ForTrain_1, label_y_ForTrain_1,
                                               input_dim_1, input_dim_2, batch_size, input_size_1, input_size_2,
                                               WEIGHTS, SAMPLE_TIMES_ForTrain, Z0_0, TRAIN_X, TRAIN_Y_ForLossCalculation,
                                               E_EPSILON_10, COV_EPSILON_10, noise_ForTrain[0], weighted_param=weighted_1,
                                               collector=collector_1
                                               )
        Opt_1 = GPyOpt.methods.BayesianOptimization(f=RegPred_ForTrain_1.fx_1,  # function to optimize
                                                    model_type=prior,
                                                    domain=bounds_1,  # box-constraints of the problem
                                                    acquisition_type=acq_func,
                                                    exact_feval=exact_f,
                                                    acquisition_optimizer_type=acq_optimizer,
                                                    kernel=kernel,
                                                    Initial_design_numdata=10
                                                    )
        Opt_1.run_optimization(BayesianOpt_iters)

        end_1 = time.time()

        FOUND_X_1 = Opt_1.x_opt
        FOUND_FX_1 = Opt_1.fx_opt
        FOUND_MEAN_1, FOUND_VAR_1 = RegPred_ForTrain_1.collector.opt_mean_var()

        print("Spent {}min, the found opt_x of 1st layer are:{} \n".format(round((end_1-start_1)/60, 1), FOUND_X_1))
        print('opt_fx is {}, opt_mean is {}, opt_var is {} \n'.format(round(FOUND_FX_1, 4),
                                                                      round(FOUND_MEAN_1, 4),
                                                                      round(FOUND_VAR_1, 4)))

        #--------------------------------- Test ---------------------------------
        noise_ForPred = generate_noise(num_layer=1, dt=dt, RandomState=RandomState_instance,
                                       expand_steps=expand_steps_ForPred,
                                       num_sub_series=serie_steps - 1, sample_times=SAMPLE_TIMES_ForPred,
                                       batch_size=batch_size, input_size_1=input_size_1, is_train=False
                                       )
        RegPred_ForPred_1 = Multi_layer_RegPred(1, RegPred_Cell_1_ForPred, g_ForPred_1, loss_mean_1_ForPred,
                                                 loss_var_1_ForPred, num_samples_ForPred_1,
                                                 label_y_ForPred_1,
                                                 input_dim_1, input_dim_2, batch_size, input_size_1, input_size_2,
                                                 WEIGHTS, SAMPLE_TIMES_ForPred, Z0_0, TRAIN_X_ForPred, TRAIN_Y,
                                                 E_EPSILON_10, COV_EPSILON_10, noise_ForPred[0],
                                                 predicted_y=pred_y_WithSigma_1_ForPred, is_train=False
                                                 )
        val_mean_1, val_var_1, val_pred_1 = RegPred_ForPred_1.fx_1_pred(FOUND_X_1)
        print("The test mean and var loss of 1st layer is:{0:.4f}".format(val_mean_1) + ", {0:.4f}\n\n".format(val_var_1))

        # save the trained result based on saved result, if it's better, then save
        my_file = Path( './experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_loss.pkl'.format(
                       expand_steps_ForTrain, kind_, ii, kind_)
                       )
        save_results_1 = True

        if os.path.isfile(my_file):
            with open(my_file, 'rb') as f:
                past_loss_1 = pkl.load(f)

            if FOUND_MEAN_1 < past_loss_1['train_loss_mean_1'] and \
                    (val_mean_1) < past_loss_1['val_loss_mean_1']:
                save_results_1 = True
            else:
                save_results_1 = False

            del past_loss_1

        if save_results_1:
            optimized_list_1.append(ii)
            # save parameters
            params_1 = {'FOUND_X_1': FOUND_X_1, 'A_bound': A_bound, 'N_bound': N_bound, 'Sigma_bound': Sigma_bound,
                        'lr_bound': lr_bound, 'phi_rho_bound': phi_rho_bound
                        }
            with open('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_params.pkl'.format(
                    expand_steps_ForTrain, kind_, ii, kind_), 'wb') as f:
                pkl.dump(params_1, f)

            describe_results = stats.describe(val_pred_1, axis=1)
            mean_ = np.squeeze(describe_results.mean)
            std_ = np.squeeze(np.sqrt(describe_results.variance))

            plt.figure()

            plt.plot(t_, np.concatenate([np.squeeze(np.exp(TRAIN_X_ForPred[0, :, -1])), mean_]), linewidth=0.5,
                     color='r')

            plt.plot(t_, np.concatenate([np.exp(TRAIN_X_ForPred[0, :]), TRAIN_Y[:, 0]]),
                     label='loss_mean:{0:.3f}'.format(val_mean_1),
                     linewidth=0.3
                     )

            plt.fill_between(t_expand_steps, np.concatenate([np.exp(TRAIN_X_ForPred[0, -1]), mean_ - 2 * std_]),
                                             np.concatenate([np.exp(TRAIN_X_ForPred[0, -1]), mean_ + 2 * std_]),
                                             color='gray', alpha=0.5)

            plt.axvline(x=serie_steps-1 - 1, color='r', linewidth=0.3)
            plt.axvline(x=serie_steps-1 + expand_steps_ForTrain - 1, color='g', linewidth=0.3)
            plt.axvline(x=serie_steps-1 + expand_steps_ForTrain + expand_steps_ForPred - 1, color='k', linewidth=0.3)
            plt.legend()

            # plt.show()
            plt.savefig('./Figure/fig_RegPred_{}/fig_{}_{}days_AvgLoss/fig_{}_{}th_1layer.eps'.format(
                        kind_, kind_, expand_steps_ForTrain, kind_, ii), format='eps')

            plt.close()

            # calculate confidential interval accuracy
            acc_confi_interval_1 = confident_interval_accuracy(TRAIN_Y[:, 0, 0], mean_ + 2 * std_, mean_ - 2 * std_)
            # acc_confi_interval_list_1.append(acc_confi_interval_1)
            print('The accuracy of confidence interval is: {}% \n'.format(acc_confi_interval_1))

            # save loss
            loss_1 = {'val_loss_mean_1': val_mean_1, 'val_loss_var_1': val_var_1,
                      'val_loss_1': val_mean_1+val_var_1,
                      'val_pred_1': val_pred_1,
                      'train_loss_mean_1':FOUND_MEAN_1,
                      'train_loss_var_1':FOUND_VAR_1,
                      'train_loss_1': FOUND_FX_1,
                      'acc_confi_interval_1':acc_confi_interval_1
                      }
            with open('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_loss.pkl'.format(
                    expand_steps_ForTrain, kind_, ii, kind_), 'wb') as f:
                pkl.dump(loss_1, f)

            print("Better! saved \n\n")

    #------------------------------------------------#
    #              Train the 2nd layer
    #------------------------------------------------#
    # train_2nd=False
    train_2nd = True

    if not train_1st:
        with open('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_params.pkl'.format(
                  expand_steps_ForTrain, kind_, ii,kind_), 'rb') as f:
            best_x_FromLayer_1 = pkl.load(f)

        FOUND_X_1 = best_x_FromLayer_1['FOUND_X_1']

    # elif train_1st and not save_results_1:
    #     with open('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_params.pkl'.format(
    #               expand_steps_ForTrain, kind_, ii,kind_), 'rb') as f:
    #         params_FromLayer_1 = pkl.load(f)
    #     A_bound = params_FromLayer_1['A_bound']
    #     N_bound = params_FromLayer_1['N_bound']
    #     Sigma_bound = params_FromLayer_1['Sigma_bound']
    #     lr_bound = params_FromLayer_1['lr_bound']
    #     phi_rho_bound = params_FromLayer_1['phi_rho_bound']

    if train_2nd:
        start_2 = time.time()
        print('time count begin for {}th, 2nd layer training'.format(ii))

        noise_ForTrain = generate_noise(num_layer=2, dt=dt, RandomState=RandomState_instance,
                                        expand_steps=expand_steps_ForTrain, num_sub_series=serie_steps - 1,
                                        sample_times=SAMPLE_TIMES_ForTrain, batch_size=batch_size,
                                        input_size_1=input_size_1, input_size_2=input_size_2, is_train=True)
        collector_2 = collector()
        RegPred_ForTrain_2 = Multi_layer_RegPred(2, RegPred_Cell_2_ForTrain, g_ForTrain_2, loss_mean_2_ForTrain,
                                                 loss_var_2_ForTrain, num_samples_ForTrain_2, label_y_ForTrain_2,
                                                 input_dim_1, input_dim_2, batch_size, input_size_1, input_size_2,
                                                 WEIGHTS, SAMPLE_TIMES_ForTrain, Z0_0, TRAIN_X, TRAIN_Y_ForLossCalculation,
                                                 E_EPSILON_10, COV_EPSILON_10, noise_ForTrain[0],
                                                 E_EPSILON_20=E_EPSILON_20, COV_EPSILON_20=COV_EPSILON_20, DW1=noise_ForTrain[1],
                                                 FOUND_X_1=FOUND_X_1, weighted_param=weighted_2, collector=collector_2
                                                 )
        Opt_2 = GPyOpt.methods.BayesianOptimization(f=RegPred_ForTrain_2.fx_2,  # function to optimize
                                                    model_type=prior,
                                                    domain=bounds_2,  # box-constraints of the problem
                                                    acquisition_type=acq_func,
                                                    exact_feval=exact_f,
                                                    acquisition_optimizer_type=acq_optimizer,
                                                    Initial_design_numdata=10
                                                    )
        Opt_2.run_optimization(BayesianOpt_iters_2)

        end_2 = time.time()

        FOUND_X_2 = Opt_2.x_opt
        FOUND_FX_2 = Opt_2.fx_opt
        FOUND_MEAN_2, FOUND_VAR_2 = RegPred_ForTrain_2.collector.opt_mean_var()

        print("Spent {}min, the found opt_x of 2nd layer are:{} \n".format(round((end_2 - start_2)/60, 1), FOUND_X_2))
        print('opt_fx is {}, opt_mean is {}, opt_var is {} \n'.format(round(FOUND_FX_2, 4),
                                                                      round(FOUND_MEAN_2, 4),
                                                                      round(FOUND_VAR_2, 4)))
        # --------------------------------- Test ---------------------------------
        noise_ForPred = generate_noise(num_layer=2, dt=dt, RandomState=RandomState_instance,
                                       expand_steps=expand_steps_ForPred,
                                       num_sub_series=serie_steps - 1, sample_times=SAMPLE_TIMES_ForPred,
                                       batch_size=batch_size, input_size_1=input_size_1, input_size_2=input_size_2,
                                       is_train=False
                                       )
        RegPred_ForPred_2 = Multi_layer_RegPred(2, RegPred_Cell_2_ForPred, g_ForPred_2, loss_mean_2_ForPred,
                                                loss_var_2_ForPred, num_samples_ForPred_2,
                                                label_y_ForPred_2,
                                                input_dim_1, input_dim_2, batch_size, input_size_1, input_size_2,
                                                WEIGHTS, SAMPLE_TIMES_ForPred, Z0_0, TRAIN_X_ForPred, TRAIN_Y,
                                                E_EPSILON_10, COV_EPSILON_10, noise_ForPred[0],
                                                predicted_y=pred_y_WithSigma_2_ForPred,
                                                E_EPSILON_20=E_EPSILON_20, COV_EPSILON_20=COV_EPSILON_20, DW1=noise_ForPred[1],
                                                FOUND_X_1=FOUND_X_1, is_train=False
                                                )
        val_mean_2, val_var_2, val_pred_2 = RegPred_ForPred_2.fx_2_pred(FOUND_X_2)
        print("The test mean and var loss of 2nd layer is:{0:.4f}".format(val_mean_2) + ", {0:.4f}\n\n".format(val_var_2))

        # save the trained result based on saved result, if it's better, then save
        my_file = Path('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_loss.pkl'.format(
            expand_steps_ForTrain, kind_, ii, kind_)
        )
        save_results = True

        if os.path.isfile(my_file):
            with open(my_file, 'rb') as f:
                past_loss_2 = pkl.load(f)

            if FOUND_MEAN_2 < past_loss_2['train_loss_mean_2'] and \
                    (val_mean_2) < past_loss_2['val_loss_mean_2']:
                save_results = True
            else:
                save_results = False

            del past_loss_2

        if save_results:
            optimized_list_2.append(ii)

            if train_1st:
                # save parameters
                params_2 = {'FOUND_X_2': FOUND_X_2, 'A_bound': A_bound, 'N_bound': N_bound, 'Sigma_bound': Sigma_bound,
                            'lr_bound': lr_bound, 'phi_rho_bound': phi_rho_bound,
                            'A_bound2': A_bound_2, 'N_bound2': N_bound_2, 'Sigma_bound2': Sigma_bound_2,
                            'lr_bound2': lr_bound_2, 'phi_rho_bound2': phi_rho_bound_2
                            }
            else:
                params_2 = {'FOUND_X_2': FOUND_X_2, 'A_bound': best_x_FromLayer_1['A_bound'], 'N_bound': best_x_FromLayer_1['N_bound'],
                            'Sigma_bound': best_x_FromLayer_1['Sigma_bound'], 'lr_bound': best_x_FromLayer_1['lr_bound'],
                            'phi_rho_bound': best_x_FromLayer_1['phi_rho_bound'],
                            'A_bound2': A_bound_2, 'N_bound2': N_bound_2, 'Sigma_bound2': Sigma_bound_2,
                            'lr_bound2': lr_bound_2, 'phi_rho_bound2': phi_rho_bound_2
                            }

            with open('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_params.pkl'.format(
                    expand_steps_ForTrain, kind_, ii, kind_), 'wb') as f:
                pkl.dump(params_2, f)

            describe_results = stats.describe(val_pred_2, axis=1)
            mean_ = np.squeeze(describe_results.mean)
            std_ = np.squeeze(np.sqrt(describe_results.variance))

            plt.figure()

            plt.plot(t_, np.concatenate([np.squeeze(np.exp(TRAIN_X_ForPred[0, :, -1])), mean_]), linewidth=0.5,
                     color='r')

            plt.plot(t_, np.concatenate([np.exp(TRAIN_X_ForPred[0, :]), TRAIN_Y[:, 0]]),
                     label='loss_mean:{0:.3f}'.format(val_mean_2),
                     linewidth=0.3
                     )

            plt.fill_between(t_expand_steps, np.concatenate([np.exp(TRAIN_X_ForPred[0, -1]), mean_ - 2 * std_]),
                             np.concatenate([np.exp(TRAIN_X_ForPred[0, -1]), mean_ + 2 * std_]),
                             color='gray', alpha=0.5)

            plt.axvline(x=serie_steps - 1 - 1, color='r', linewidth=0.3)
            plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain - 1, color='g', linewidth=0.3)
            plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1, color='k', linewidth=0.3)
            plt.legend()

            # plt.show()
            plt.savefig('./Figure/fig_RegPred_{}/fig_{}_{}days_AvgLoss/fig_{}_{}th_2layer.eps'.format(
                kind_, kind_, expand_steps_ForTrain, kind_, ii), format='eps')

            plt.close()

            # calculate confidential interval accuracy
            acc_confi_interval_2 = confident_interval_accuracy(TRAIN_Y[:, 0, 0], mean_ + 2 * std_, mean_ - 2 * std_)
            # acc_confi_interval_list_1.append(acc_confi_interval_1)
            print('The accuracy of confidence interval is: {}% \n\n'.format(acc_confi_interval_2))

            # save loss
            loss_2 = {'val_loss_mean_2': val_mean_2, 'val_loss_var_2': val_var_2,
                      'val_loss_2': val_mean_2 + val_var_2,
                      'val_pred_2': val_pred_2,
                      'train_loss_mean_2': FOUND_MEAN_2,
                      'train_loss_var_2': FOUND_VAR_2,
                      'train_loss_2': FOUND_FX_2,
                      'acc_confi_interval_2': acc_confi_interval_2,
                      }
            with open('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_loss.pkl'.format(
                    expand_steps_ForTrain, kind_, ii, kind_), 'wb') as f:
                pkl.dump(loss_2, f)

            print("Better! saved \n")



