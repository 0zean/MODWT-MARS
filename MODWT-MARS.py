from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from pyearth import Earth
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load MRA detail and scale coefficients
modwt_mra = pd.read_csv('modwt_mra.csv')

d1, d2, d3, s3 = modwt_mra["d1"], modwt_mra["d2"], modwt_mra["d3"], modwt_mra["s3"]

# Get stock data over same period as MRA
with open("key.txt") as f:
    key = f.readline().strip("\n")

ts = TimeSeries(key=key, output_format='pandas')
data, meta_data = ts.get_daily_adjusted("SPY", outputsize="full")
data = data.loc[data.index >= "2009-01-02"]

price = data['4. close'].iloc[::-1]
returns = np.log(price).diff().dropna()


# Get price labels using same AR scheme
series = returns

D = 4  # number of regressors
T = 1  # delay
N = len(series)
data = np.zeros((N - T - (D - 1) * T, D))
lbls = np.zeros((N - T - (D - 1) * T,))

for t in range((D - 1) * T, N - T):
    data[t - (D - 1) * T, :] = [series[t - 3 * T], series[t - 2 * T], series[t - T], series[t]]
    lbls[t - (D - 1) * T] = series[t + T]
trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]


def MODWT_MARS(series, regressors=4, delay=1):
    series = series.values

    D = regressors  # number of regressors
    T = delay  # delay
    N = len(series)
    data = np.zeros((N - T - (D - 1) * T, D))
    lbls = np.zeros((N - T - (D - 1) * T,))

    for t in range((D - 1) * T, N - T):
        data[t - (D - 1) * T, :] = [series[t - 3 * T], series[t - 2 * T], series[t - T], series[t]]
        lbls[t - (D - 1) * T] = series[t + T]

    trnData = data[:lbls.size - round(lbls.size * 0.3), :]
    trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
    chkData = data[lbls.size - round(lbls.size * 0.3):, :]
    chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]

    aa = np.array(chkLbls[-4:]).reshape(1, -1)
    chkData = np.append(chkData, aa, axis=0)

    mars = Earth()
    mars.fit(trnData, trnLbls)
    boosted_mars = AdaBoostRegressor(estimator=mars, n_estimators=50, learning_rate=0.1, loss='exponential')
    boosted_mars.fit(trnData, trnLbls)
    oos_preds = boosted_mars.predict(chkData)

    return oos_preds


# MRA
D1_test = pd.DataFrame(MODWT_MARS(d1), columns=["D1"])
D2_test = pd.DataFrame(MODWT_MARS(d2), columns=["D2"])
D3_test = pd.DataFrame(MODWT_MARS(d3), columns=["D3"])
S3_test = pd.DataFrame(MODWT_MARS(s3), columns=["S3"])

D1_test = pd.concat([D1_test, D2_test], axis=1)
D1_test = pd.concat([D1_test, D3_test], axis=1)
D1_test = pd.concat([D1_test, S3_test], axis=1)

D1_test['sum'] = D1_test.sum(axis=1)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - actual[:-1])).astype(int))


plt.plot(np.array(D1_test["sum"]), color="g")
plt.plot(chkLbls)
plt.annotate("Mean Directional Accuracy = {:.3f}".format(mda(chkLbls, D1_test["sum"][:-1])), (-20, max(chkLbls)+0.01))
plt.annotate("RMSE = {:.3e}".format(mean_squared_error(chkLbls, D1_test["sum"][:-1], squared=False)), (-20, max(chkLbls)))
plt.legend(["Pred", "Actual"])

figure = plt.gcf()
figure.set_size_inches(8, 6)
plt.savefig("Results/test.png", dpi=252)
plt.show()
