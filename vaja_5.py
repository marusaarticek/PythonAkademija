# Multipla Linearna Regresija
# napoved vrednosti in korelacije med več spremenljivkami/atributi
# 1. Sestavi zasnovno matriko: vrstice podatkovne vrednosti, stolpci parametri
# 2. Izracun regresijskih koeficientov:
#		a.Množenje transponirane zasnovne matrike same s sabo.
#		b.Množenje transponirane zasnovne matrike z vektorjem ciljnih vrednosti.
#		c.Množenje inverzne matrike iz koraka a z matriko iz koraka b.
# 3. S koeficienti izracun napovedanih ciljnih vrednosti. Razlike med opazovanimi
# 	 in napovedanimi vrednostmi-ostanki.
# 4. Preizkus modela.

import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Definirani atributi, ki jih preskocimo
columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

#Izbrani podatki
df = pd.read_csv('naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

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
print(c, list(zip(x, m)))

y_pred_train = lr.predict(x_test)

#print("Prediction for test set: {}".format(y_pred_train))

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_train})
print(mlr_diff.head())

#Model Evaluation
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error

meanAbPerErr = mean_absolute_percentage_error(y_test, y_pred_train )
meanAbErr = metrics.mean_absolute_error(y_test,y_pred_train)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_train)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_train))
print('R squared: {:.2f}'.format(lr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
print('mean_absolute_percentage_error(y_test, predictions)', meanAbPerErr)
#print(df.corr())

#https://sefidian.com/2022/06/18/a-guide-on-regression-error-metrics-with-python-code/
#https://machinelearningmastery.com/feature-selection-for-regression-data/