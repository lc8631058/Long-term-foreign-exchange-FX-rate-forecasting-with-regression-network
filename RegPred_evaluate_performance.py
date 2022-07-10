import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from metrics import *
from sklearn.metrics import r2_score

def plot_pred_without_label(serie_steps, expand_steps_ForTrain, expand_steps_ForPred, line_ForPlot,
                            UpperBound_ForPlot, LowerBound_ForPlot, loss, save_address):
    t_expand_steps = np.linspace((serie_steps - 1 + expand_steps_ForTrain) - 1,
                                 serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1,
                                 expand_steps_ForPred + 1
                                 )
    t_ = np.linspace(0, serie_steps - 1 + + expand_steps_ForTrain + expand_steps_ForPred - 1,
                     serie_steps - 1 + + expand_steps_ForTrain + expand_steps_ForPred)

    plt.figure()

    plt.plot(t_, line_ForPlot,
             label=' train_fx:{0:.4f}; val_fx:{0:.4f}'.format(loss['train_loss_1'],
                                                              loss['val_loss_WithSigma_1']),
             linewidth=0.3
             )
    plt.fill_between(t_expand_steps, UpperBound_ForPlot, LowerBound_ForPlot,
                     color='gray', alpha=0.5)

    plt.axvline(x=serie_steps - 1 - 1, color='r', linewidth=0.3)
    plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain - 1, color='g', linewidth=0.3)
    plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1, color='k', linewidth=0.3)
    plt.legend()

    plt.show()
    plt.savefig(save_address, format='eps')
    plt.close()
    return None

def load_saved_results(path_params=None, path_loss=None, path_fig=None, import_img=False):
    """
    load the saved results from RegPred Net
    :param path_params: path of params
    :param path_loss: path of loss
    :param path_fig: path of fig
    :return: loaded data
    """
    with open(path_params, 'rb') as f:
        params =  pkl.load(f)
    with open(path_loss, 'rb') as f:
        loss =  pkl.load(f)
    if import_img:
        return params, loss, mpimg.imread(path_fig)
    else:
        return params, loss

def get_stats(y_pred):
    """
    Get statistic results like mean and std
    :param y_pred:
    :return:
    """
    describe_results = stats.describe(y_pred, axis=1)
    mean_ = np.squeeze(describe_results.mean)
    std_ = np.squeeze(np.sqrt(describe_results.variance))
    return (mean_.astype(np.float64)).round(4), (std_.astype(np.float64)).round(4)

kind_ = 'GBP'
serie_steps = 1830
expand_steps_ForTrain = 200 # the steps you wanna decoder to expand
expand_steps_ForPred = 100
num_years = 5
# random_state = 1
# RandomState_ = np.random.RandomState(seed=random_state)
# noise = RandomState_.normal(loc=0.0, scale=1.0, size=expand_steps_ForPred)

path = "./Data/RegPred_dataset/RegPred_{}Y".format(num_years)
## load input x
# with open(path + '/RegPred_X_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
#     data_ = pkl.load(f)
# # load loss x for loss calculation
# with open(path + '/RegPred_lossX_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
#     data_LossCalculation = pkl.load(f)
# load corresponding labels
with open(path + '/RegPred_futureX_{}_{}Y_{}daysLossCal.pkl'.format(kind_, num_years, expand_steps_ForTrain), 'rb') as f:
    labels_ = np.squeeze(pkl.load(f), axis=-1)

#--__--__--__--__--__--__--__--__--__--__--__--__--__--__#
#                   Open saved results
#--__--__--__--__--__--__--__--__--__--__--__--__--__--__#
R_squared_1, R_squared_2 = [], []
MDA_1, MDA_2 = [], []
val_mean_loss_list_1, val_mean_loss_list_2, val_mean_loss_list_3 = [], [], []
val_mean_loss_list_OnlyBest = []
RMSE_1, RMSE_2, RMSE_3, RMSE_OnlyBest = [], [], [], []
MAPE_1, MAPE_2, MAPE_3, MAPE_OnlyBest = [], [], [], []
R_1, R_2, R_3, R_OnlyBest = [], [], [], []
SpearmanR_1, SpearmanR_2, SpearmanR_3, SpearmanR_OnlyBest = [], [], [], []
KendallTau_1, KendallTau_2, KendallTau_3, KendallTau_OnlyBest = [], [], [], []
# TheilU_1, TheilU_2, TheilU_3 = [], [], []
# MAE_1, MAE_2, MAE_3 = [], [], []
# AbsDev_1, AbsDev_2, AbsDev_3 = [], [], []
# MASE_1, MASE_2, MASE_3 = [], [], []
# DTW_1, DTW_2, DTW_3 = [], [], []
evaluate_performance=True
# evaluate_performance=False
# which=27
if evaluate_performance:

    for ii in range(1, labels_.shape[0]+1):
    # for ii in range(which, which + 1):
        # -_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
        #                           layer 1
        # -_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
        params_1, loss_1 = \
            load_saved_results('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_params.pkl'.
                                format(expand_steps_ForTrain, kind_, ii, kind_),
                               './experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_1layer_loss.pkl'.
                               format(expand_steps_ForTrain, kind_, ii, kind_)
                               # './Figure/fig_RegPred_{}/fig_{}_{}days/fig_{}_{}th_1layer.eps'.
                               # format(kind_, kind_, expand_steps_ForTrain, kind_, ii)
                               )
        mean_1, std_1 = get_stats(loss_1['val_pred_1'])

        # MAPE, R, TheilU
        R_squared_1.append(r2_score(labels_[ii-1], mean_1))
        MDA_1.append(MDA(mean_1, labels_[ii-1]))
        RMSE_1.append(RMSE(mean_1, labels_[ii-1]))
        # MAPE_1.append(MAPE(mean_1, labels_[ii-1]))
        R_1.append(R(mean_1, labels_[ii-1]))
        # SpearmanR_1.append(SpearmanR(mean_1, labels_[ii-1]))
        # KendallTau_1.append(KendallTau(mean_1, labels_[ii-1]))
        # DTW_1.append(DTW(mean_1, labels_[ii-1]))
        # TheilU_1.append(TheilU(mean_1, labels_[ii-1]))
        # MAE_1.append(MAE(mean_1, labels_[ii-1]))
        # AbsDev_1.append(AbsDev(mean_1, labels_[ii-1]))
        # MASE_1.append(MASE(data_[ii-1], mean_1, labels_[ii-1]))

        # RMSE
        val_mean_loss_list_1.append(loss_1['val_loss_mean_1'])

        # -_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
        #                           layer 2
        # -_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
        params_2, loss_2 = \
            load_saved_results('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_params.pkl'.
                                format(expand_steps_ForTrain, kind_, ii, kind_),
                               './experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_2layer_loss.pkl'.
                               format(expand_steps_ForTrain, kind_, ii, kind_)
                               # './Figure/fig_RegPred_{}/fig_{}_{}days/fig_{}_{}th_2layers.eps'.
                               # format(kind_, kind_, expand_steps_ForTrain, kind_, ii)
                               )
        mean_2, std_2 = get_stats(loss_2['val_pred_2'])

        # MAPE, R, TheilU
        R_squared_2.append(r2_score(labels_[ii - 1], mean_2))
        MDA_2.append(MDA(mean_2, labels_[ii - 1]))
        RMSE_2.append(RMSE(mean_2, labels_[ii - 1]))
        R_2.append(R(mean_2, labels_[ii - 1]))

        # MAPE_2.append(MAPE(mean_2, labels_[ii-1]))
        # SpearmanR_2.append(SpearmanR(mean_2, labels_[ii - 1]))
        # KendallTau_2.append(KendallTau(mean_2, labels_[ii - 1]))
        # DTW_2.append(DTW(mean_2, labels_[ii - 1]))
        # TheilU_2.append(TheilU(mean_2, labels_[ii-1]))
        # MAE_2.append(MAE(mean_2, labels_[ii - 1]))
        # AbsDev_2.append(AbsDev(mean_2, labels_[ii - 1]))
        # MASE_2.append(MASE(np.squeeze(data_[ii - 1]), mean_2, labels_[ii - 1]))

        # RMSE
        val_mean_loss_list_2.append(loss_2['val_loss_mean_2'])

        # # -_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
        # #                           layer 3
        # # -_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
        # params_3, loss_3 = \
        #     load_saved_results('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_3layer_params.pkl'.
        #                         format(expand_steps_ForTrain, kind_, ii, kind_),
        #                        './experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_3layer_loss.pkl'.
        #                        format(expand_steps_ForTrain, kind_, ii, kind_)
        #                        # './Figure/fig_RegPred_{}/fig_{}_{}days/fig_{}_{}th_3layers.eps'.
        #                        # format(kind_, kind_, expand_steps_ForTrain, kind_, ii)
        #                        )
        # mean_3, std_3 = get_stats(loss_3['val_pred_y_ForPred_3'])
        #
        # # MAPE, R, TheilU
        # RMSE_3.append(RMSE(mean_3, labels_[ii - 1]))
        # MAPE_3.append(MAPE(mean_3, labels_[ii-1]))
        # R_3.append(R(mean_3, labels_[ii-1]))
        # SpearmanR_3.append(SpearmanR(mean_3, labels_[ii - 1]))
        # KendallTau_3.append(KendallTau(mean_3, labels_[ii - 1]))
        # # DTW_3.append(DTW(mean_3, labels_[ii - 1]))
        # # TheilU_3.append(TheilU(mean_3, labels_[ii-1]))
        # # MAE_3.append(MAE(mean_3, labels_[ii - 1]))
        # # AbsDev_3.append(AbsDev(mean_3, labels_[ii - 1]))
        # # MASE_3.append(MASE(np.squeeze(data_[ii - 1]), mean_3, labels_[ii - 1]))
        #
        # # RMSE
        # val_mean_loss_list_3.append(loss_3['val_loss_mean_3'])

        # min_over_layers = np.argmin([-loss_1['train_loss_1'],
        #                              -loss_2['train_loss_2'],
        #                              -loss_3['train_loss_3']])

        # loss_agg = [loss_1['val_loss_mean_1'], loss_2['val_loss_mean_2'], loss_3['val_loss_mean_3']]
        # val_mean_loss_list_OnlyBest.append(loss_agg[min_over_layers])
        #
        # # based on smallest train loss to choose mean
        # mean_agg = [mean_1, mean_2, mean_3]
        # RMSE_OnlyBest.append(RMSE(mean_agg[min_over_layers], labels_[ii - 1]))
        # MAPE_OnlyBest.append(MAPE(mean_agg[min_over_layers], labels_[ii - 1]))
        # R_OnlyBest.append(R(mean_agg[min_over_layers], labels_[ii - 1]))
        # SpearmanR_OnlyBest.append(SpearmanR(mean_agg[min_over_layers], labels_[ii - 1]))
        # KendallTau_OnlyBest.append(KendallTau(mean_agg[min_over_layers], labels_[ii - 1]))


    print("Time series: {} \n".format(kind_))
    print("the mean loss of indiscriminate layers is: {} \n".format(np.mean(val_mean_loss_list_OnlyBest))
          )
    print("Average R of layer-1: \n avg.{} / med.{} / std.{} \n".format(np.mean(R_1), np.median(R_1), np.std(R_1)) +
          "R of layer-2: \n avg.{} / med.{} / std.{}\n".format(np.mean(R_2), np.median(R_2), np.std(R_2))
          )
    print("Average R-squared of layer-1: \n avg.{} / med.{} / std.{} \n".format(np.mean(R_squared_1), np.median(R_squared_1), np.std(R_squared_1)) +
          "R-squared of layer-2: \n avg.{} / med.{} / std.{} \n".format(np.mean(R_squared_2), np.median(R_squared_2), np.std(R_squared_2))
          )
    print("Average RMSE of layer-1: \n avg.{} / med.{} / std.{} \n".format(np.mean(RMSE_1), np.median(RMSE_1), np.std(RMSE_1)) +
          "RMSE of layer-2: \n avg.{} / med.{} / std.{} \n".format(np.mean(RMSE_2), np.median(RMSE_2), np.std(RMSE_2))
          )
    print("Average MDA of layer-1: \n avg.{} / med.{} / std.{} \n".format(np.mean(MDA_1), np.median(MDA_1),
                                                                           np.std(MDA_1)) +
          "MDA of layer-2: \n avg.{} / med.{} / std.{} \n".format(np.mean(MDA_1), np.median(MDA_1), np.std(MDA_1))
          )
    # print("Average SpearmanR of layer-1: \n avg.{} / med.{} \n".format(np.mean(SpearmanR_1), np.median(SpearmanR_1)) +
    #       "SpearmanR of layer-2: \n avg.{} / med.{} \n".format(np.mean(SpearmanR_2), np.median(SpearmanR_2))
    #       # "SpearmanR of layer-3: \n avg.{} / med.{} \n".format(np.mean(SpearmanR_3), np.median(SpearmanR_3)) +
    #       # "SpearmanR of Only Best: \n avg.{} / med.{} \n".format(np.mean(SpearmanR_OnlyBest), np.median(SpearmanR_OnlyBest))
    #       )
    # print("Average KendallTau of layer-1: \n avg.{} / med.{} \n".format(np.mean(KendallTau_1), np.median(KendallTau_1)) +
    #       "KendallTau of layer-2: \n avg.{} / med.{} \n".format(np.mean(KendallTau_2), np.median(KendallTau_2))
    #       # "KendallTau of layer-3: \n avg.{} / med.{} \n".format(np.mean(KendallTau_3), np.median(KendallTau_3)) +
    #       # "KendallTau of Only Best: \n avg.{} / med.{} \n".format(np.mean(KendallTau_OnlyBest),
    #       #                                                        np.median(KendallTau_OnlyBest))
    #       )
    # print("Average MAPE of layer-1: \n avg.{} / med.{} \n".format(np.mean(MAPE_1), np.median(MAPE_1)) +
    #       "MAPE of layer-2: \n avg.{} / med.{} \n".format(np.mean(MAPE_2), np.median(MAPE_2))
    #       # "MAPE of layer-3: \n avg.{} / med.{} \n".format(np.mean(MAPE_3), np.median(MAPE_3)) +
    #       # "MAPE of Only Best: \n avg.{} / med.{} \n".format(np.mean(MAPE_OnlyBest), np.median(MAPE_OnlyBest))
    #       )
    # "Average DTW of layer-1 to layer-3 are: \n avg.{} / med.{},\n avg.{} / med.{},\n avg.{} / med.{}, respectively \n".
    # format(np.mean(DTW_1), np.median(DTW_1),
    #        np.mean(DTW_2), np.median(DTW_2),
    #        np.mean(DTW_3), np.median(DTW_3)) +
    # "Average TheilU of layer-1 to layer-3 are: \n avg.{} / med.{},\n avg.{} / med.{},\n avg.{} / med.{}, respectively \n".
    # format(np.mean(TheilU_1), np.median(TheilU_1),
    #        np.mean(TheilU_2), np.median(TheilU_2),
    #        np.mean(TheilU_3), np.median(TheilU_3)) +
    # "Average MAE of layer-1 to layer-3 are: \n avg.{} / med.{},\n avg.{} / med.{},\n avg.{} / med.{}, respectively \n".
    # format(np.mean(MAE_1), np.median(MAE_1),
    #        np.mean(MAE_2), np.median(MAE_2),
    #        np.mean(MAE_3), np.median(MAE_3)) +
    # "Average AbsDev of layer-1 to layer-3 are: \n avg.{} / med.{},\n avg.{} / med.{},\n avg.{} / med.{}, respectively \n".
    # format(np.mean(AbsDev_1), np.median(AbsDev_1),
    #        np.mean(AbsDev_2), np.median(AbsDev_2),
    #        np.mean(AbsDev_3), np.median(AbsDev_3)) +
    # "Average MASE of layer-1 to layer-3 are: \n avg.{} / med.{},\n avg.{} / med.{},\n avg.{} / med.{}, respectively \n".
    # format(np.mean(MASE_1), np.median(MASE_1),
    #        np.mean(MASE_2), np.median(MASE_2),
    #        np.mean(MASE_3), np.median(MASE_3))

#-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_--_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
#                                               Plot for paper
#-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_--_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-#
kind_2 = 'GBP'
abspath = '/Users/lilinwei/Desktop/Reinforcement Learning/Implementation/gym-FX-Liquidation/LSTM-parameter regression/'
path = "./Data/RegPred_dataset/RegPred_{}Y".format(num_years)
# load input x
with open(path + '/RegPred_X_{}_{}Y_{}daysLossCal.pkl'.format(kind_2, num_years, expand_steps_ForTrain), 'rb') as f:
    data_2 = pkl.load(f)
# load loss x for loss calculation
with open(path + '/RegPred_lossX_{}_{}Y_{}daysLossCal.pkl'.format(kind_2, num_years, expand_steps_ForTrain), 'rb') as f:
    data_LossCalculation_2 = pkl.load(f)
# load corresponding labels
with open(path + '/RegPred_futureX_{}_{}Y_{}daysLossCal.pkl'.format(kind_2, num_years, expand_steps_ForTrain), 'rb') as f:
    labels_2 = np.squeeze(pkl.load(f), axis=-1)

# CNY [2, 19, 27, 20, 83]
# GBP [12, 59, 83]
# plot_for_paper=True
plot_for_paper=False
if plot_for_paper:
    layer_idx = 2
    fig_idx = [13, 48, 87]
    # fig_idx = np.arange(1, len(data_2)+1)

    for ii in fig_idx:
        # read the record on second layer
        params_2, loss_2 = \
            load_saved_results('./experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_{}layer_params.pkl'.
                               format(expand_steps_ForTrain, kind_2, ii, kind_2, layer_idx),
                               './experimental_results/experimental_results_{}days_AvgLoss/{}/{}th_{}_{}layer_loss.pkl'.
                               format(expand_steps_ForTrain, kind_2, ii, kind_2, layer_idx),
                               './Figure/fig_RegPred_{}/fig_{}_{}days_AvgLoss/fig_{}_{}th_{}layers.eps'.
                               format(kind_2, kind_2, expand_steps_ForTrain, kind_2, ii, layer_idx)
                               )
        mean_2, std_2 = get_stats(loss_2['val_pred_2'])

        t_expand_steps = np.linspace((serie_steps - 1 + expand_steps_ForTrain) - 1,
                                     serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1,
                                     expand_steps_ForPred + 1
                                     )
        t_input = np.linspace(0 + 1300, 1830, 529)
        t_loss_cal = np.linspace(1831, 1831+expand_steps_ForTrain, expand_steps_ForTrain)

        plt.figure()

        # plot input
        plt.plot(t_input, data_2[ii - 1][1 + 1300:].squeeze(), label='Input', linewidth=0.8, color='k')
        # plot loss cal
        plt.plot(t_loss_cal, data_LossCalculation_2[ii - 1].squeeze(), label='Label_Train', linewidth=0.8, color='darkcyan')
        plt.plot(t_expand_steps[:-1], labels_2[ii - 1], label='Label_Pred', linewidth=0.8, color='blue')


        # plot mean
        plt.plot(t_expand_steps, np.concatenate([data_LossCalculation_2[ii - 1][-1], mean_2]),
                 label='Pred', color='r',
                 linewidth=1.5)

        # plt.plot([], [], ' ', label="R: {0:.3f}".format(R(mean_2, labels_2[ii - 1])))
        # plt.plot([], [], ' ', label="RMSE: {0:.3f}".format(RMSE(mean_2, labels_2[ii - 1])))
        # plt.plot([], [], ' ', label="MAPE: {0:.3f}%".format(MAPE(mean_2, labels_2[ii - 1])*100))

        plt.fill_between(t_expand_steps,
                         np.concatenate([data_LossCalculation_2[ii - 1][-1], mean_2 + 2 * std_2]),
                         np.concatenate([data_LossCalculation_2[ii - 1][-1], mean_2 - 2 * std_2]),
                         color='gray', alpha=0.4)

        plt.axvline(x=serie_steps - 1 - 1, color='blue', linestyle='--', linewidth=1.)
        plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain - 1, linestyle='--', color='red', linewidth=1.)
        plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1, linestyle='--',
                    color='k', linewidth=1.)

        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)

        plt.xlabel('Time (days)', fontsize=15)
        plt.ylabel('FX rate', fontsize=15)
        plt.legend(prop={'size':10.5})
        plt.title("R: {0:.3f},  ".format(R(mean_2, labels_2[ii - 1])) +
                  "RMSE: {0:.3f},  ".format(RMSE(mean_2, labels_2[ii - 1])) +
                  "MDA: {0:.3f}".format(MDA(mean_2, labels_2[ii - 1])), fontsize=15
                  )
        # plt.show()
        fig_save_address = abspath + 'Figure/Plot_for_paper/RegPred/{}/RegPred_fig_2layers_{}_{}th.pdf'.format(kind_2, kind_2, ii)
        plt.savefig(fig_save_address, format='pdf', bbox_inches='tight')
        plt.close()


# fig_idx = [29, 47, 83]
# layer_idx = 2
#
# for ii in fig_idx:
#     # read the record on second layer
#     params_2, loss_2, fig_2 = \
#         load_saved_results('./experimental_results/experimental_results_{}days/{}/{}th_{}_{}layer_params.pkl'.
#                            format(expand_steps_ForTrain, kind_2, ii, kind_2, layer_idx),
#                            './experimental_results/experimental_results_{}days/{}/{}th_{}_{}layer_loss.pkl'.
#                            format(expand_steps_ForTrain, kind_2, ii, kind_2, layer_idx),
#                            './Figure/fig_RegPred_{}/fig_{}_{}days/fig_{}_{}th_{}layers.eps'.
#                            format(kind_2, kind_2, expand_steps_ForTrain, kind_2, ii, layer_idx)
#                            )
#     mean_2, std_2 = get_stats(loss_2['val_pred_y_ForPred_2'])
#
#     t_expand_steps = np.linspace(200, 300-1, 100)
#     t_ = np.linspace(0, 300-1, 300)
#
#     plt.figure()
#     # plot label
#
#     plt.plot(t_, np.concatenate([data_LossCalculation_2[ii - 1].squeeze(),
#                                  labels_2[ii - 1]]),
#              label='Label',
#              linewidth=0.8
#              )
#
#     for jj in range(50):
#
#         # plot mean
#         # plt.plot(t_expand_steps, np.concatenate([data_LossCalculation_2[ii - 1][-1], mean_2]),
#         #          label='Pred', color='r',
#         #          linewidth=1.2)
#         plt.plot(t_expand_steps, np.squeeze(loss_2['val_pred_y_ForPred_2'])[:, jj],
#                  linewidth=0.6)
#
#         # plt.fill_between(t_expand_steps,
#         #                  np.concatenate([data_LossCalculation_2[ii - 1][-1], mean_2 + 2 * std_2]),
#         #                  np.concatenate([data_LossCalculation_2[ii - 1][-1], mean_2 - 2 * std_2]),
#         #                  color='gray', alpha=0.4)
#
#         # plt.axvline(x=serie_steps - 1 - 1, color='darkred', linestyle='--', linewidth=0.8)
#         # plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain - 1, linestyle='--', color='g', linewidth=0.8)
#         # plt.axvline(x=serie_steps - 1 + expand_steps_ForTrain + expand_steps_ForPred - 1, linestyle='--',
#         #             color='k', linewidth=0.8)
#
#     plt.plot(t_expand_steps, mean_2, color='r', label='Pred',
#              linewidth=0.8)
#     plt.xlabel('Time (days)', fontsize=13)
#     plt.ylabel('FX rate', fontsize=13)
#     plt.legend()
#
#     plt.show()
#     fig_save_address = './Figure/Plot_for_paper/RegPred_fig_{}layers_{}_{}th.pdf'.format(layer_idx, kind_2, ii)
#     plt.savefig(fig_save_address, format='pdf', bbox_inches='tight')