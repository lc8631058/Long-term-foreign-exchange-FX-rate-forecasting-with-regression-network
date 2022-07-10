import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle as pkl
import pandas as pd
def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)

# plt.plot(x, y)

def posterior(optimizer, x_obs, y_obs, grid):
    """
    calculate posterior   
    """
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    """
    plot gaussian process 
    """
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    # fig.suptitle(
    #     'Gaussian Process and Utility Function After {} Steps'.format(steps),
    #     fontdict={'size': 30}
    # )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=2, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=7, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=0.2, fc='c', ec='None', label='95% confidence interval', color='blue')

    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Acquisition Function', color='blue')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next x', markerfacecolor='red', markeredgecolor='r', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Acquisition', fontdict={'size': 18})
    acq.set_xlabel('x', fontdict={'size': 18})

    axis.legend(loc=2, bbox_to_anchor=(0.75, 0.98), borderaxespad=0., prop={'size': 13})
    acq.legend(loc=2, bbox_to_anchor=(0.76, 0.95), borderaxespad=0., prop={'size': 13})
    # plt.show()
    # ax.set_rasterized(True)
    plt.savefig("./Figure/BayOpt_ForPaper.pdf")

# plot_BayOpt = True
plot_BayOpt = False
if plot_BayOpt:
    optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=27)

    for ii in range(14):
        optimizer.maximize(init_points=2, n_iter=0, kappa=5)

    plot_gp(optimizer, x, y)

# plot_full_FX = True
plot_full_FX = False
if plot_full_FX:
    # load full serie
    data_path = "./Data/calibration parameters/data_FX_dayly.pkl"
    with open(data_path, "rb") as f:
        data = pkl.load(f)

    CNY = np.array(data['EURCNY_Curncy'])
    USD = np.array(data['EURUSD_Curncy'])
    GBP = np.array(data['EURGBP_Curncy'])

    t_ = np.linspace(0, len(CNY)+1, len(CNY))
    for ii_, (curve_, type_, color) in enumerate(zip([CNY, USD, GBP], ['EUR/CNY', 'EUR/USD', 'EUR/GBP'], ['tab:blue', 'tab:blue', 'tab:blue'])):
        plt.figure()
        plt.plot(t_, curve_, label=type_, linewidth=0.8, color=color)
        # plt.plot(t_, np.log(curve_), label=type_, linewidth=0.8, color=color)

        plt.xlabel('Time(days)', fontsize=18)
        plt.ylabel('FX rate', fontsize=18)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.xlim(0, 4975)
        plt.legend(prop={'size': 16})
        plt.savefig('./Figure/FX_rate_{}.pdf'.format(ii_), bbox_inches='tight')
    # plt.show()


from utils import delta_brownian
plot_Wiener = False
# plot_Wiener = True
if plot_Wiener:
    brownian = delta_brownian(1, 1, 500)

    t_ = np.linspace(0, 500, 500)
    plt.figure()
    plt.plot(t_, brownian[0, :], label='Wiener process', linewidth=1, color='tab:blue')
    # plt.plot([], [], ' ', label=u"Δt = 1")
    # plt.title(u"Δt = 1", fontsize=20)
    plt.xlabel('Time step', fontsize=18)
    plt.ylabel('Sampling value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xlim(0, 4975)
    plt.legend(prop={'size': 15})
    plt.savefig('./Figure/Wiener.pdf', bbox_inches='tight')
    # plt.show()

    Xt_1 = 7.0
    sigma = 0.01
    mu = 0.001
    Xt_list = [Xt_1]
    for ii in range(499):
        Xt = Xt_1 + sigma * brownian[0, ii] + mu
        Xt_list.append(Xt)
        Xt_1 = Xt
    plt.figure()
    plt.plot(t_, Xt_list, label='Brownian motion', linewidth=1, color='tab:blue')
    # plt.plot([], [], ' ', label=r"Δt=1, $\sigma$=0.01, n=10")
    plt.title(r"$X_0$={}, $\sigma$={}, $\mu$={}".format(7.0, sigma, mu), fontsize=20)
    plt.xlabel('Time step', fontsize=18)
    plt.ylabel('Sampling value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 15})
    plt.savefig('./Figure/Brownian.pdf', bbox_inches='tight')
    # plt.show()

    Xt_1 = 7.0
    alpha = 0.02
    sigma = 0.01
    n=7.3
    Xt_list = [Xt_1]
    for ii in range(499):
        Xt = Xt_1 + alpha * (n-Xt_1) + sigma * brownian[0, ii]
        Xt_list.append(Xt)
        Xt_1 = Xt

    plt.figure()
    plt.plot(t_, Xt_list, label='Mean-reverting', linewidth=1.3, color='tab:blue')
    # plt.plot([], [], ' ', label=r"Δt=1, $\sigma$=0.01, n=10, $\alpha$=0.05")
    plt.title(r"$X_0$={}, $\alpha$={}, $\sigma$={}, $n$={}".format( 7.0, alpha, sigma, n), fontsize=20)
    plt.xlabel('Time step', fontsize=18)
    plt.ylabel('Sampling value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 15})
    plt.savefig('./Figure/MeanReverting.pdf', bbox_inches='tight')
    # plt.show()

    Xt_1 = np.log(8.533)
    A = -0.01893642
    N = 0.03792842
    SIGMA = 0.00709833
    Xt_list = [Xt_1]
    for ii in range(499):
        Xt = Xt_1 + A * Xt_1 + N + SIGMA * brownian[0, ii]
        Xt_list.append(Xt)
        Xt_1 = Xt

    plt.figure()
    plt.plot(t_, np.exp(Xt_list), label='Generalized OU process', linewidth=1.3, color='tab:blue')
    # plt.plot([], [], ' ', label=r"Δt=1, $\sigma$=0.01, n=10, $\alpha$=0.05")
    plt.title(r"$X_0$={:.3f}, A={:.3f}, N={:.3f}, $\Sigma$={:.3f}".format(np.exp(Xt_1), A, N, SIGMA), fontsize=18)
    plt.xlabel('Time step', fontsize=18)
    plt.ylabel('Sampling value', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 15})
    plt.savefig('./Figure/GOU.pdf', bbox_inches='tight')
    # plt.show()


    
