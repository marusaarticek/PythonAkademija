import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Definirani atributi, ki jih preskocimo
columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

#Izbrani podatki
df = pd.read_csv('Naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

# Izberi podatke za x in y
x = df.drop(columns = 'MAX_VREDNOST')
y=df['MAX_VREDNOST']

# Razdelimo podatke na učno in testno množico 7:3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 100)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)

c = lr.intercept_

m = lr.coef_
#print(c, m)

y_pred_train = lr.predict(x_test)

#print("Prediction for test set: {}".format(y_pred_train))

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_train})
print(mlr_diff.head())

#Model Evaluation

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

new_list = list[[meanAbErr], [meanPerErr], [rootMeanSqErr], [smape_]]
print(new_list)


# Kako jih dodati skupaj?

data = {'mae': [meanAbErr],
        'mape': [meanPerErr],
        'rmse': [rootMeanSqErr],
        'smape': [smape_]}

df_r = pd.read_csv('Rezultat.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)
df_r['MAE'], df_r['MAPE'],df_r['RMSE'],df_r['SMAPE'] =  [[meanAbErr], [meanPerErr], [rootMeanSqErr], [smape_]]
print(df_r)