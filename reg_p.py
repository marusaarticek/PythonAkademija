import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

x = df.drop(columns = 'MAX_VREDNOST')
y=df['MAX_VREDNOST']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)


#Linear Regression
x1 = sm.add_constant(x_train)
ols = sm.OLS(y_train,x1)
lr = ols.fit()

selected_features = list(x.columns)
pmax = 1
while (len(selected_features)>0):
    p= []
    x_1 = x[selected_features]
    x_1 = sm.add_constant(x_1)
    model = sm.OLS(y,x_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = selected_features)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        selected_features.remove(feature_with_p_max)
    else:
        break  
        
print('The selected features are :', selected_features)

columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

#Model z najbolsimi vrednostmi
a = df[['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']]
b=df['MAX_VREDNOST']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(a, b)

#print('Intercept: \n', regr.intercept_)
#print('Coefficients: \n', regr.coef_)
c = []
c.append(regr.coef_)
print(c)
# with statsmodels
a = sm.add_constant(a) # adding a constant
 
model = sm.OLS(b, a).fit()
predictions = model.predict(a) 
 
print_model = model.summary()
print(print_model)

import csv 

fields = ['PRA_DAN', 'PRA_PRED', 'BDP1', 'BDP3', 'URA_VEC', 'MIN_PRET_TED', 'TEMP_MAX', 'MAX_VRED_LM1', 'PMIN_VRED_LM1', 'PMAX_VRED_LM1']
rows = c

# name of csv file 
filename = "Rezultat.csv"
    
# writing to csv file 
with open(filename, 'w', newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)