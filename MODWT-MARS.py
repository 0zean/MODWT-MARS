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

modwt_mra = pd.read_csv('modwt_mra.csv')

D1, D2, D3, S3 = modwt_mra['D1'], modwt_mra['D2'], modwt_mra['D3'], modwt_mra['S3']

ticker = 'SPY'
stock = pdr.get_data_yahoo(ticker.upper(), start='2009-01-01', end=str(datetime.now())[0:11])
stock = stock.Close
returns = np.log(stock).diff().dropna()


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

# Establish Autoregressive array
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
D1_train = pd.DataFrame(MODWT_MARS_TRAIN(D1))
D2_train = pd.DataFrame(MODWT_MARS_TRAIN(D2))
D3_train = pd.DataFrame(MODWT_MARS_TRAIN(D3))
S3_train = pd.DataFrame(MODWT_MARS_TRAIN(S3))

D1_train = D1_train.rename(columns={0: 'D1'})
D2_train = D2_train.rename(columns={0: 'D2'})
D3_train = D3_train.rename(columns={0: 'D3'})
S3_train = S3_train.rename(columns={0: 'S3'})

D1_train = pd.concat([D1_train, D2_train], axis=1)
D1_train = pd.concat([D1_train, D3_train], axis=1)
D1_train = pd.concat([D1_train, S3_train], axis=1)

D1_train['sum'] = D1_train.sum(axis=1)
plt.plot(np.array(D1_train['sum']), color='g')
plt.plot(trnLbls)
	
	
DA_train = pd.DataFrame(trnLbls)
DA_train['com'] = DA_train[0].shift(1)
DA_train['ACC'] = DA_train[0] - DA_train['com']
DA_train['ACC'] = DA_train['ACC'].mask(DA_train['ACC'] > 0 , 1)
DA_train['ACC'] = DA_train['ACC'].mask(DA_train['ACC'] < 0 , 0)

DA_pred = pd.DataFrame(D1_train['sum'])
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
D1_test = pd.DataFrame(MODWT_MARS_TEST(D1))
D2_test = pd.DataFrame(MODWT_MARS_TEST(D2))
D3_test = pd.DataFrame(MODWT_MARS_TEST(D3))
S3_test = pd.DataFrame(MODWT_MARS_TEST(S3))

D1_test = D1_test.rename(columns={0: 'D1'})
D2_test = D2_test.rename(columns={0: 'D2'})
D3_test = D3_test.rename(columns={0: 'D3'})
S3_test = S3_test.rename(columns={0: 'S3'})

D1_test = pd.concat([D1_test, D2_test], axis=1)
D1_test = pd.concat([D1_test, D3_test], axis=1)
D1_test = pd.concat([D1_test, S3_test], axis=1)

D1_test['sum'] = D1_test.sum(axis=1)
plt.plot(np.array(D1_test['sum']), color='g')
plt.plot(chkLbls)
plt.legend(['Pred','Actual'])
plt.savefig("Results/test.jpg")
	
	
DA_test = pd.DataFrame(chkLbls)
DA_test['com'] = DA_test[0].shift(1)
DA_test['ACC'] = DA_test[0] - DA_test['com']
DA_test['ACC'] = DA_test['ACC'].mask(DA_test['ACC'] > 0 , 1)
DA_test['ACC'] = DA_test['ACC'].mask(DA_test['ACC'] < 0 , 0)

DA_pred = pd.DataFrame(D1_test['sum'])
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

pred = D1_test['sum']
pred = pred[:-1]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print('MSE: ' + str(mean_squared_error(chkLbls, pred)))
print('RMSE: ' + str(rmse(np.array(pred), chkLbls)))
print('R Squared: ' + str(r2_score(chkLbls, pred)))

Direction = D1_test['sum']

if Direction[449] > Direction[448]:
    print("UP")
else:
    print("DOWN")

previous = stock[-1:]
ret = np.array(Direction)
ret = ret[-1:]
price = np.array((np.exp(ret)) * previous)
print('Price: ' + str(np.round(price[-1:], 2)))
