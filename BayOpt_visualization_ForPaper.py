# from bayes_opt import BayesianOptimization
# from bayes_opt import UtilityFunction
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
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
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

# plot_optimization=True
plot_optimization=False

if plot_optimization:
    layer1 = [951931.0, 1e+30, 17.924880981445312, 67.10111236572266, 1e+30, 111.00079345703125, 4.387814044952393,
              1e+30, 77.68949127197266, 834437185536.0, 12.49703311920166, 1e+30, 9.077191352844238, 11.294689178466797,
              1e+30, 8.433246612548828, 111.52271270751953, 1e+30, 4.903690338134766, 3.9105138778686523, 49.38916778564453,
              4.914067268371582, 4.794996738433838, 1e+30, 6.639663219451904, 3.7460274696350098, 10.844828605651855,
              3.1738555431365967, 4.794229507446289, 71.0706787109375, 3.862802028656006, 11.744473457336426, 30.383407592773438,
              22.134729385375977, 7.850053310394287, 1e+30, 6.151491165161133, 11.579919815063477, 7090.392578125, 30.294212341308594,
              66.35419464111328, 4.906482219696045, 5.741688251495361, 16.128673553466797, 4.545529842376709, 3.58715558052063,
              7.890362739562988, 4.277482986450195, 8.86074161529541, 10.46957015991211, 5.485141277313232, 1e+30, 3.2535316944122314,
              3.585207939147949, 4.2774834632873535, 10.277660369873047, 3.6208391189575195, 5.687417984008789, 4.656127452850342,
              2.3951010704040527, 80.65293884277344, 15.25363540649414, 3.4182748794555664, 3.101811408996582, 152.76904296875,
              3.8132548332214355, 1e+30, 2.801410436630249, 1e+30, 2.8792428970336914, 3.942732334136963, 4.013376235961914,
              1.9052119255065918, 35.726985931396484, 36.27806091308594, 2.6535401344299316, 2.087264060974121, 2.3084490299224854,
              3.296358108520508, 7.564485549926758, 2.440570831298828, 2.8238308429718018, 60.852848052978516, 4.944131374359131,
              4.116124629974365, 2.938657760620117, 4.542359352111816, 40.867210388183594, 6.0790815353393555, 3.1484179496765137,
              3.417614221572876, 4.435369968414307, 671517.4375, 3.033215045928955, 4.531558036804199, 26.294401168823242,
              38210.1484375, 3.2179105281829834, 467183008.0, 2011.92919921875, 12.046697616577148, 4.426353454589844, 5.371927261352539,
              149.94717407226562, 6.82676362991333, 2.1933934688568115, 1.5630943775177002, 2.279695987701416, 2.396362543106079, 47738966016.0
              , 11384.2890625, 3.2773168087005615, 13.536821365356445, 15.197895050048828, 2.472386121749878, 4.565345764160156, 14.588135719299316,
              10025.458984375, 1e+30, 2.394029378890991, 113.05181121826172, 3.36562180519104, 3.5407652854919434, 48.38614273071289,
              6.687837600708008, 3.765326976776123, 16.79001808166504, 2.7846646308898926, 2.7767348289489746, 11.210618019104004,
              31.034868240356445, 61.39493179321289, 2.6913905143737793, 1e+30, 3.216588020324707, 1e+30, 3.807206869125366, 5.399643421173096
              , 4.574002265930176, 4.277482986450195, 4.692232131958008, 13.59068489074707, 4.277482986450195, 4.225809574127197,
              3.3088531494140625, 7.080433368682861, 19.562007904052734, 1e+30, 6.159873962402344, 12.00675106048584, 3.207563877105713,
              70.08019256591797, 3.1433470249176025, 1e+30, 7.797388076782227, 5.6728129386901855, 11.89302921295166,2.4928672313690186,1.759520411491394,
              171.84852600097656, 39.45547866821289,2.6012792587280273, 22.88678550720215, 3.842085599899292, 2.6717638969421387,
              2.1178369522094727, 2.3209519386291504, 1.896546721458435, 24.37904167175293, 4.815711975097656, 3.2646353244781494, 22.88678550720215
              , 3.0946319103240967, 19.702342987060547, 6.9645185470581055, 1.5316219329833984, 3.0251951217651367,320.24578857421875,3.5312159061431885,
              6.809319019317627, 1.9975461959838867, 6.807830810546875, 3.500016689300537, 3.655892848968506, 3.5079123973846436, 1e+30, 1.9498176574707031,
              25.883399963378906, 4.164849281311035, 3.504343032836914, 5.638332366943359, 1e+30, 8.040868759155273, 19.537532806396484,
              5.1667046546936035, 2.6873159408569336, 5.59617805480957, 3.2864081859588623, 1.4437005519866943, 22.886695861816406, 2.1178369522094727,
              22.88678550720215, 3.1060853004455566, 3.2819981575012207, 5.135546684265137
              ]
    #
    # for ii in layer1:
    #     if ii == 1e+30:

    # report = pd.read_csv('./experimental_results/experimental_results_200days_AvgLoss/CNY/83th_CNY_1layer_report.csv',
    #                      sep=' ')
