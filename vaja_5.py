# Multiple Linear regression
# napoved vrednosti in korelacije med več spremenljivkami
# oziroma atributi

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

# Definirani atributi, ki jih preskocimo
columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('naloga.csv', delimiter=";", header=0, usecols=lambda x: x not in columns_to_skip)
 
df['MAX_VREDNOST'] = df['MAX_VREDNOST'].apply(lambda x : float(x.replace(",",".")))
df['MAX_PRET_TED'] = df['MAX_PRET_TED'].apply(lambda x : float(x.replace(",",".")))


x = df['MAX_PRET_TED']# independent variable
y = df['MAX_VREDNOST']# dependent variable


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)
