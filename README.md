# MODWT-MARS
Hybrid MODWT-MARS model for financial time series forecasting

A hybrid model that combines maximal overlap discrete wavelet transform multiresolution analysis with multivariate adaptive regression splines to give improved one-step-ahead forecasting.

This hyrbid model was inspired from this paper by Jothimani, D., Shankar, R., Yadav, S.S.:
[Discrete Wavelet Transform Based Prediction of Stock Index: A Study on National Stock Exchange FiftyIndex](https://arxiv.org/ftp/arxiv/papers/1605/1605.07278.pdf)

The premise of this model is to decompose the log-return of the closing price of a stock using MODWT multiresolution analysis and use the detail coefficients and smooth coefficient as inputs for the MARS model.

MODWT allows for decomposition of nondyadic lengthed signals and is circular-shift invariant making it suitable for financial time series prediction. It decomposes a signal in time-frequency space, exposing time-frequency information in the orignal signal not seen otherwise.
Since the MODWT-MRA phase shifts the coefficients to be approximately time-aligned it is used. 

A level 3 MODWT-MRA, using a Dauchecies least asymmetric wavelet (symlet), is performed in R using reflection of the right boundary to eliminate edge effects*. A feature set is generated autoregressively by taking the 4 previous values at each time step given by the expression below.

At time t, value x is defined as: 

![equation](http://latex.codecogs.com/gif.latex?x(t)%3Df(x(t-1),x(t-2),x(t-3),x(t-4))) 

This is done for the 3 detail coefficients D1, D2, D3 and the smooth coefficient S3.

Each of these are used as input for the MARS model, where Adaboost and bagging regressor boosters are used and then combined with extreme gradient boosting. The predicted coeffcient vectors are then summed up afterwards to produce the final prediction vector.

* It is a common problem amongst quantitative finance papers to incorrectly apply wavelet decompositions to financial data and use these decompositions as features for model training. Both the DWT and MODWT make assumptions of the original signal, namely that it is periodic on the given domain and that all points on the signal are known beforehand. This is not the case for financial data especially in real-time situations. This model produces results that are even better than the listed paper's but it is misleading since the return series is not periodic and the choice of symlet looks ahead in the transform. More complex methods are required to actually apply the wavelet transform in this application such as Segmented Lifting wavelet transform or a two-stage decomposition scheme to eliminate the boundary effects that make the prediction at the boundary useless.

**Requirments**
- numpy
- pandas
- statsmodels
- pandas datareader
- datetime
- matplotlib
- scipy
- xgboost
- [py-earth](https://github.com/scikit-learn-contrib/py-earth)
- sklearn



**In Sample Results**:
![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/train.jpg)

**In Sample Results** (Zoomed in):
![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/train_zoom.jpg)

**Directional Accuracy on predicted training set**:
![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/DA_train.jpg)

**Out of Sample Results**:
![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/test.jpg)

**Out of Sample Results** (Zoomed in, With one day ahead forecast):
![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/test_zoom.jpg)

**Directional Accuracy on predicted test set with error metrics**:

![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/DA_test.jpg)
