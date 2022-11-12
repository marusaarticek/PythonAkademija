import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU' , 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('Naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

from datetime import datetime

df['DATUM'] = pd.to_datetime(df['DATUM'])
  
test=df[df['DATUM'].dt.month == 12] #test
train=df[df['DATUM'].dt.month != 12]

x_train = train.drop(columns = ['MAX_VREDNOST', 'DATUM'])
x_test=test.drop(columns = ['MAX_VREDNOST', 'DATUM'])
y_train=train['MAX_VREDNOST']
y_test=test['MAX_VREDNOST'] 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#c = lr.intercept_
#m = lr.coef_
#print(c, m)

y_pred_train = lr.predict(x_test)
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_train})
print(mlr_diff.head())

#Errors
# 1. Training set...


from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

meanAbErr = metrics.mean_absolute_error(y_test,y_pred_train)
meanPerErr = metrics.mean_absolute_percentage_error(y_test, y_pred_train)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_train))
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
smape_ = smape(y_test,y_pred_train)

print('SMAPE:', smape(y_test,y_pred_train))
print('MAE: Mean Absolute Error:', meanAbErr)
print('RMSE: Root Mean Square Error:', rootMeanSqErr)
print('MAPE:', meanPerErr)

dff = pd.DataFrame(
    {   
        'mae': [meanAbErr],
        'mape': [meanPerErr],
        'rmse': [rootMeanSqErr],
        'smape': [smape_]
    }
)

dff.to_csv("NapakeTestno.csv", sep='\t')
