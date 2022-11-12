#   ---- Multipla Linearna Regresija ----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

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

lr = LinearRegression()
lr.fit(x_train, y_train)

c = lr.intercept_
m = lr.coef_

y_pred = lr.predict(x_test)

