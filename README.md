# MODWT-MARS
Hybrid MODWT-MARS model for finance time series forecasting

A hybrid model that combines maximal overlap discrete wavelet transform multiresolution analysis with multivariate adaptive regression splines to give improved one-step-ahead forecasting.

The concept of this hyrbid model was inspired from this paper by Jothimani, D., Shankar, R., Yadav, S.S.:

[Discrete Wavelet Transform Based Prediction of Stock Index: A Study on National Stock Exchange FiftyIndex](https://arxiv.org/ftp/arxiv/papers/1605/1605.07278.pdf)


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

**Out of Sample Results** (Zoomed in):
![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/test_zoom.jpg)

**Directional Accuracy on predicted test set with error metrics**:

![alt text](https://github.com/Nicholas-Picini/MODWT-MARS/blob/master/Results/DA_test.jpg)
