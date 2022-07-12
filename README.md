I am currently busy with my thesis, then I will optimize the code. 

# Long-term-foreign-exchange-FX-rate-forecasting-with-regression-network

This repository is about the realization of the article:
L. Li, P.-A. Matt, and C. Heumann, “Forecasting foreign exchange rates with regression networks tuned by bayesian optimization”, Engineering Applications of Artificial Intelligence, 2022.

Network architecture: (More details please refer to the article)
<img width="1080" alt="image" src="https://user-images.githubusercontent.com/25768931/178138629-3aec5e33-eda5-451d-b319-b1a7034bf9c5.png">

Some of the results visualization:
<img width="896" alt="image" src="https://user-images.githubusercontent.com/25768931/178138670-8fcc698a-ce71-4b4a-aa60-8ee2605b81ee.png">

## Abstract

The article is concerned with the problem of multi-step financial time series forecasting of Foreign Exchange (FX) rates. To address this problem, we introduce a regression network termed
RegPred Net. The exchange rate to forecast is treated as a stochastic process. It is assumed
to follow a generalization of Brownian motion and the mean-reverting process referred to as
generalized Ornstein-Uhlenbeck (OU) process, with time-dependent coefficients. Using past
observed values of the input time series, these coefficients can be regressed online by the cells
of the first half of the network (Reg). The regressed coefficients depend only on - but are very
sensitive to - a small number of hyperparameters required to be set by a global optimization
procedure for which, Bayesian optimization is an adequate heuristic. Thanks to its multi-layered
architecture, the second half of the regression network (Pred) can project time-dependent values
for the OU process coefficients and generate realistic trajectories of the time series. Predictions
can be easily derived in the form of expected values estimated by averaging values obtained by
Monte Carlo simulation. The forecasting accuracy on a 100 days horizon is evaluated for several of the most important FX rates such as EUR/USD, EUR/CNY and EUR/GBP. Our experimental results show that the RegPred Net significantly outperforms ARMA, ARIMA, LSTMs,
and Autoencoder-LSTM models in terms of metrics measuring the absolute error (RMSE) and
correlation between predicted and actual values (Pearson’s R, R-squared, MDA). Compared to
black-box deep learning models such as LSTM, RegPred Net has better interpretability, simpler
structure, and fewer parameters. In addition, it can predict dynamic parameters that reflect trends
in exchange rates over time, which provides decision-makers with important information when
dealing with sequential decision-making tasks.

## Getting Started

### Dependencies

* Tensorflow 1.14.0 , scikit-learn 0.24.2, numpy 1.19.2
* Mac OS

### Installing

* The algorithm introduced in our paper is implemented in RegPred_GpyOpt.py

<!-- ### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
``` -->
<!-- 
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->

## Authors

Linwei Li

## Version History

<!-- * 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]() -->
* 0.1
    * Initial Release

## License

<!-- This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details -->

## Acknowledgments

<!-- Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
 -->
