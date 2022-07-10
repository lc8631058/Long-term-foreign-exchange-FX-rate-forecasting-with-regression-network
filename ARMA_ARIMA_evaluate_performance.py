from statsmodels.tsa.arima_model import ARIMA, ARMA
import pickle as pkl
from pandas import datetime
from matplotlib import pyplot as plt
from metrics import *

def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

# import pyramid
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import r2_score

Auto_arima=True
# Auto_arima=False
if Auto_arima:
    # #############################################################################
    # load data
    kind_ = 'GBP'
    num_years = 5
    serie_steps = 1830
    expand_steps_ForTrain = 200
    expand_steps_ForPred = 100
    path = "./Data/RegPred_dataset/RegPred_{}Y".format(num_years)

    # load input x
    with open(path + '/RegPred_X_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        data_ = pkl.load(f)
    # load loss x for loss calculation
    with open(path + '/RegPred_lossX_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        data_LossCalculation = pkl.load(f)
    # load corresponding labels
    with open(path + '/RegPred_futureX_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        labels_ = pkl.load(f)

    train_x = np.concatenate([np.array(data_), np.array(data_LossCalculation)], axis=1).squeeze()
    train_x = train_x[:, 1:] # 1830-1
    train_y = np.array(labels_).squeeze()

    R_2, MDA_, R_, SpearmanR_, KendallTau_, RMSE_, MAPE_ = [], [], [], [], [], [], []
    orders = []
    predictions = []
    # model_name = 'ARIMA'
    # order_ARIMA = (5, 1, 1)
    # order_ARIMA = (2, 0, 2)

    for ii in range(train_x.shape[0]):
        history_ = train_x[ii]
        true_ = train_y[ii]

        # Fit a simple auto_arima model
        # modl = pyramid.arima.ARIMA(order_ARIMA, suppress_warnings=True)
        # modl.fit(history_)
        # pred_y, conf_int = modl.predict(n_periods=expand_steps_ForPred,  return_conf_int=True)

        modl = auto_arima(history_, start_p=0, start_q=0,
                          max_p=20, max_q=20, seasonal=True,
                          stepwise=True, suppress_warnings=True,
                          error_action='ignore',
                          d=0, max_d=20)

        orders.append(modl.get_params()['order'])
        # Create predictions for the future, evaluate on test
        pred_y, conf_int = modl.predict(n_periods=expand_steps_ForPred, return_conf_int=True)

        R_2.append(r2_score(true_, pred_y))
        MDA_.append(MDA(pred_y, true_))
        R_.append(R(pred_y, true_))
        # SpearmanR_.append(SpearmanR(pred_y, true_))
        # KendallTau_.append(KendallTau(pred_y, true_))
        RMSE_.append(RMSE(pred_y, true_))
        # MAPE_.append(MAPE(pred_y, true_))
        predictions.append(pred_y)

    print('R: mean {} / median {} / std {}'.format(np.mean(R_), np.median(R_), np.std(R_)))
    print('R-sqaured: mean {} / median {} / std {}'.format(np.mean(R_2), np.median(R_2), np.std(R_2)))
    print('RMSE: mean {} / median {} / std {}'.format(np.mean(RMSE_), np.median(RMSE_), np.std(RMSE_)))
    print('MDA: mean {} / median {} / std {}'.format(np.mean(MDA_), np.median(MDA_), np.std(MDA_)))
    # print('MAPE: mean {} / median {} / std {}'.format(np.mean(MAPE_), np.median(MAPE_)))

#-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_--_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
#                                               Evaluate performance
#-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_--_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
# evaluate_performance=True
evaluate_performance=False
if evaluate_performance:

    # load data
    kind_ = 'GBP'
    num_years = 5
    serie_steps = 1830
    expand_steps_ForTrain = 200
    expand_steps_ForPred = 100
    path = "./Data/RegPred_dataset/RegPred_{}Y".format(num_years)

    # load input x
    with open(path + '/RegPred_X_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        data_ = pkl.load(f)
    # load loss x for loss calculation
    with open(path + '/RegPred_lossX_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        data_LossCalculation = pkl.load(f)
    # load corresponding labels
    with open(path + '/RegPred_futureX_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
        labels_ = pkl.load(f)

    train_x = np.concatenate([np.array(data_), np.array(data_LossCalculation)], axis=1).squeeze()
    train_x = train_x[:, 1:] # 1830-1
    train_y = np.array(labels_).squeeze()

    R_, SpearmanR_, KendallTau_, RMSE_, MAPE_ = [], [], [], [], []

    predictions = []
    model_name = 'ARIMA'
    order_ARIMA = (5, 1, 1)
    order_ARMA = (2, 0)
    for ii in range(train_x.shape[0]):
        history_ = train_x[ii]
        true_ = train_y[ii]

        if model_name == 'ARIMA':
            model = ARIMA(history_, order=order_ARIMA)
        elif model_name == 'ARMA':
            model = ARMA(history_, order=order_ARMA)

        model_fit = model.fit(disp=0, transparams=False)

        output = model_fit.forecast(expand_steps_ForPred)
        pred_y = output[0]

        R_.append(R(pred_y, true_))
        SpearmanR_.append(SpearmanR(pred_y, true_))
        KendallTau_.append(KendallTau(pred_y, true_))
        RMSE_.append(RMSE(pred_y, true_))
        MAPE_.append(MAPE(pred_y, true_))
        predictions.append(pred_y)

    print('R: mean {} / std {}'.format(np.mean(R_), np.std(R_)))
    print('SpearmanR: mean {} / std {}'.format(np.mean(SpearmanR_), np.std(SpearmanR_)))
    print('KendallTau: mean {} / std {}'.format(np.mean(KendallTau_), np.std(KendallTau_)))
    print('RMSE: mean {} / std {}'.format(np.mean(RMSE_), np.std(RMSE_)))
    print('MAPE: mean {} / std {}'.format(np.mean(MAPE_), np.std(MAPE_)))


#-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_--_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
#                                               Plot for paper
#-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_--_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
plot_for_paper=True
# plot_for_paper=False
kind_3 = 'GBP'
abspath = '/Users/lilinwei/Desktop/Reinforcement Learning/Implementation/gym-FX-Liquidation/LSTM-parameter regression/'
if plot_for_paper:
    num_years = 5
    serie_steps = 1830
    expand_steps_ForTrain = 200
    expand_steps_ForPred = 100
    path = "./Data/RegPred_dataset/RegPred_{}Y".format(num_years)

    # load input x
    with open(path + '/RegPred_X_{}_{}Y_{}daysLossCal.pkl'.format(kind_3, num_years, expand_steps_ForTrain), 'rb') as f:
        data_2 = pkl.load(f)
    # load loss x for loss calculation
    with open(path + '/RegPred_lossX_{}_{}Y_{}daysLossCal.pkl'.format(kind_3, num_years, expand_steps_ForTrain), 'rb') as f:
        data_LossCalculation_2 = pkl.load(f)
    # load corresponding labels
    with open(path + '/RegPred_futureX_{}_{}Y_{}daysLossCal.pkl'.format(kind_3, num_years, expand_steps_ForTrain), 'rb') as f:
        labels_2 = np.squeeze(pkl.load(f), axis=-1)

    train_x = np.concatenate([np.array(data_2), np.array(data_LossCalculation_2)], axis=1).squeeze()
    # train_x = train_x[:, 1 +1000:] # 1830-1
    train_y = np.array(labels_2).squeeze()


    layer_idx = 2
    fig_idx = [13, 48, 87]
    # fig_idx = np.arange(1, len(data_2) + 1)
    # fig_idx = [78]
    model_name = 'ARIMA'
    # order_ARIMA = (5, 1, 1)
    # order_ARMA = (3, 3)

    for ii in fig_idx:
        history_ = train_x[ii-1]
        true_ = train_y[ii-1]

        modl = auto_arima(history_, start_p=0, start_q=0,
                          max_p=20, max_q=20, seasonal=True,
                          stepwise=True, suppress_warnings=True,
                          error_action='ignore',
                          d=0, max_d=20)

        pred_y, conf_int = modl.predict(n_periods=expand_steps_ForPred, return_conf_int=True)

        # if model_name == 'ARIMA':
        #     model = ARIMA(history_, order=order_ARIMA)
        # elif model_name == 'ARMA':
        #     model = ARMA(history_, order=order_ARMA)

        # model_fit = model.fit(disp=0)

        # output = model_fit.forecast(expand_steps_ForPred)
        # pred_y = output[0]

        # modl = pyramid.arima.ARIMA(order_ARIMA, suppress_warnings=True)
        # modl.fit(history_)
        # pred_y, conf_int = modl.predict(n_periods=expand_steps_ForPred, return_conf_int=True)

        t_expand_steps = np.linspace((serie_steps - 1 + expand_steps_ForTrain) - 1,
                                     serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1,
                                     expand_steps_ForPred + 1
                                     )
        t_input = np.linspace(0 + 1300, 1830, 529)
        t_loss_cal = np.linspace(1831, 1831 + expand_steps_ForTrain, expand_steps_ForTrain)

        plt.figure()

        # plot input
        plt.plot(t_input, data_2[ii - 1][1 + 1300:].squeeze(), label='Input', linewidth=0.8, color='k')
        # plot loss cal
        plt.plot(t_loss_cal, data_LossCalculation_2[ii - 1].squeeze(), label='Label_Train', linewidth=0.8,
                 color='darkcyan')
        plt.plot(t_expand_steps[:-1], labels_2[ii - 1], label='Label_Pred', linewidth=0.8, color='blue')

        # plot mean
        plt.plot(t_expand_steps, np.concatenate([data_LossCalculation_2[ii - 1][-1], pred_y]),
                 label='Pred', color='r',
                 linewidth=1.5)

        plt.fill_between(t_expand_steps,
                         np.concatenate([data_LossCalculation_2[ii - 1][-1], conf_int[:, 0]]),
                         np.concatenate([data_LossCalculation_2[ii - 1][-1], conf_int[:, 1]]),
                         color='gray', alpha=0.4)

        plt.axvline(x=serie_steps - 1 - 1, color='blue', linestyle='--', linewidth=1.)
        plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain - 1, linestyle='--', color='red', linewidth=1.)
        plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1, linestyle='--',
                    color='k', linewidth=1.)

        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        plt.xlabel('Time (days)', fontsize=15)
        plt.ylabel('FX rate', fontsize=15)
        plt.legend(prop={'size': 10.5})
        plt.title("R: {0:.3f},  ".format(R(pred_y, labels_2[ii - 1])) +
                  "RMSE: {0:.3f},  ".format(RMSE(pred_y, labels_2[ii - 1])) +
                  "MDA: {0:.3f}".format(MDA(pred_y, labels_2[ii - 1])), fontsize=15
                  )
        # plt.show()
        fig_save_address = abspath + 'Figure/Plot_for_paper/ARIMA/{}/ARIMA_{}_fig_{}th.pdf'.format(kind_3, kind_3, ii)
        plt.savefig(fig_save_address, format='pdf', bbox_inches='tight')
        plt.close()