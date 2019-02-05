import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from pyearth import Earth
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb

modwt = pd.read_csv('modwt.csv')
modwt_mra = pd.read_csv('modwt_mra.csv')

del modwt['Unnamed: 0']
del modwt_mra['Unnamed: 0']

D1, D2, D3, S3 = modwt['D1'], modwt['D2'], modwt['D3'], modwt['S3']
W1, W2, W3, V3 = modwt_mra['W1'], modwt_mra['W2'], modwt_mra['W3'], modwt_mra['V3']

ticker = 'SPY'
stock = pdr.get_data_yahoo(ticker.upper(), start='2009-01-01', end=str(datetime.now())[0:11])
stock = stock.Close
returns = np.log(stock).diff().dropna()


# Stationarity test
result = adfuller(V3.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


def MODWT_MARS_TRAIN(series, regressors=4, delay=1, N=2000):
	series = series[len(series)-N:]
	series = np.array(series)
	series = series.reshape(-1, 1)

	D = regressors  # number of regressors
	T = delay  # delay
	N = N
	series = series[500:]
	data = np.zeros((N - 500 - T - (D - 1) * T, D))
	lbls = np.zeros((N - 500 - T - (D - 1) * T,))

	for t in range((D - 1) * T, N - 500 - T):
		data[t - (D - 1) * T, :] = [series[t - 3 * T], series[t - 2 * T], series[t - T], series[t]]
		lbls[t - (D - 1) * T] = series[t + T]
	trnData = data[:lbls.size - round(lbls.size * 0.3), :]
	trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]

	mars = Earth()
	mars.fit(trnData, trnLbls)
	boosted_mars = AdaBoostRegressor(base_estimator=mars, n_estimators=25, learning_rate=0.01, loss='exponential')
	boosted_mars.fit(trnData, trnLbls)
	preds = boosted_mars.predict(trnData)

	return preds


series = returns[len(returns)-2000:]
series = np.array(series)
series = series.reshape(-1, 1)

D = 4
T = 1
N = 2000
series = series[500:]
lbls = np.zeros((N - 500 - T - (D - 1) * T,))

for t in range((D - 1) * T, N - 500 - T):
	lbls[t - (D - 1) * T] = series[t + T]
trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]


# MRA
W1_train = pd.DataFrame(MODWT_MARS_TRAIN(W1))
W2_train = pd.DataFrame(MODWT_MARS_TRAIN(W2))
W3_train = pd.DataFrame(MODWT_MARS_TRAIN(W3))
V3_train = pd.DataFrame(MODWT_MARS_TRAIN(V3))

W1_train = W1_train.rename(columns={0: 'W1'})
W2_train = W2_train.rename(columns={0: 'W2'})
W3_train = W3_train.rename(columns={0: 'W3'})
V3_train = V3_train.rename(columns={0: 'V3'})

W1_train = pd.concat([W1_train, W2_train], axis=1)
W1_train = pd.concat([W1_train, W3_train], axis=1)
W1_train = pd.concat([W1_train, V3_train], axis=1)

W1_train['sum'] = W1_train.sum(axis=1)
plt.plot(np.array(W1_train['sum']), color='g')
plt.plot(trnLbls)
	
	
DA_train = pd.DataFrame(trnLbls)
DA_train['com'] = DA_train[0].shift(1)
DA_train['ACC'] = DA_train[0] - DA_train['com']
DA_train['ACC'] = DA_train['ACC'].mask(DA_train['ACC'] > 0 , 1)
DA_train['ACC'] = DA_train['ACC'].mask(DA_train['ACC'] < 0 , 0)

DA_pred = pd.DataFrame(W1_train['sum'])
DA_pred['com'] = DA_pred['sum'].shift(1)
DA_pred['ACC2'] = DA_pred['sum'] - DA_pred['com']
DA_pred['ACC2'] = DA_pred['ACC2'].mask(DA_pred['ACC2'] > 0 , 1)
DA_pred['ACC2'] = DA_pred['ACC2'].mask(DA_pred['ACC2'] < 0 , 0)

DA = pd.DataFrame(DA_train['ACC'])
DA = DA.join(DA_pred['ACC2'])
DA['score'] = 0
DA['score'] = DA['score'].mask(DA['ACC'] == DA['ACC2'], 1)

AC = DA['score'].value_counts()
ACC = round((AC[1] / len(trnLbls)) * 100, 3)

print('Directional Accuracy: ' + str(ACC) + ' %')


def MODWT_MARS_TEST(series, regressors=4, delay=1, N=2000):
    series = series[len(series)-2000:]
    series = np.array(series)
    series = series.reshape(-1, 1)

    D = regressors  # number of regressors
    T = delay  # delay
    N = N
    series = series[500:]
    data = np.zeros((N - 500 - T - (D - 1) * T, D))
    lbls = np.zeros((N - 500 - T - (D - 1) * T,))

    for t in range((D - 1) * T, N - 500 - T):
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
    boosted_mars = AdaBoostRegressor(base_estimator=mars, n_estimators=50, learning_rate=0.1, loss='exponential')
    bag = BaggingRegressor(base_estimator=mars, n_estimators=50)
    bag.fit(trnData, trnLbls)
    boosted_mars.fit(trnData, trnLbls)
    pred2 = bag.predict(chkData)
    oos_preds = boosted_mars.predict(chkData)
    
    stack_predict = np.vstack([oos_preds, pred2]).T
    
    params_xgd = {
            'max_depth': 7,
            'objective': 'reg:linear',
            'learning_rate': 0.05,
            'n_estimators': 10000
            }
    clf = xgb.XGBRegressor(**params_xgd)
    clf.fit(stack_predict[:-1,:], chkLbls, eval_set=[(stack_predict[:-1,:], chkLbls)], 
        eval_metric='rmse', early_stopping_rounds=20, verbose=False)

    xgb_pred = clf.predict(stack_predict)

    return xgb_pred


# MRA
W1_test = pd.DataFrame(MODWT_MARS_TEST(W1))
W2_test = pd.DataFrame(MODWT_MARS_TEST(W2))
W3_test = pd.DataFrame(MODWT_MARS_TEST(W3))
V3_test = pd.DataFrame(MODWT_MARS_TEST(V3))

W1_test = W1_test.rename(columns={0: 'W1'})
W2_test = W2_test.rename(columns={0: 'W2'})
W3_test = W3_test.rename(columns={0: 'W3'})
V3_test = V3_test.rename(columns={0: 'V3'})

W1_test = pd.concat([W1_test, W2_test], axis=1)
W1_test = pd.concat([W1_test, W3_test], axis=1)
W1_test = pd.concat([W1_test, V3_test], axis=1)

W1_test['sum'] = W1_test.sum(axis=1)
plt.plot(np.array(W1_test['sum']), color='g')
plt.plot(chkLbls)
plt.legend(['Pred','Actual'])
	
	
DA_test = pd.DataFrame(chkLbls)
DA_test['com'] = DA_test[0].shift(1)
DA_test['ACC'] = DA_test[0] - DA_test['com']
DA_test['ACC'] = DA_test['ACC'].mask(DA_test['ACC'] > 0 , 1)
DA_test['ACC'] = DA_test['ACC'].mask(DA_test['ACC'] < 0 , 0)

DA_pred = pd.DataFrame(W1_test['sum'])
DA_pred['com'] = DA_pred['sum'].shift(1)
DA_pred['ACC2'] = DA_pred['sum'] - DA_pred['com']
DA_pred['ACC2'] = DA_pred['ACC2'].mask(DA_pred['ACC2'] > 0 , 1)
DA_pred['ACC2'] = DA_pred['ACC2'].mask(DA_pred['ACC2'] < 0 , 0)

DA = pd.DataFrame(DA_test['ACC'])
DA = DA.join(DA_pred['ACC2'])
DA['score'] = 0
DA['score'] = DA['score'].mask(DA['ACC'] == DA['ACC2'], 1)

AC = DA['score'].value_counts()
ACC = round((AC[1] / len(chkLbls)) * 100, 3)

print('Directional Accuracy: ' + str(ACC) + ' %')

pred = W1_test['sum']
pred = pred[:-1]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print('MSE: ' + str(mean_squared_error(chkLbls, pred)))
print('RMSE: ' + str(rmse(np.array(pred), chkLbls)))
print('R Squared: ' + str(r2_score(chkLbls, pred)))

Direction = W1_test['sum']

if Direction[449] > Direction[448]:
    print("UP")
else:
    print("DOWN")
