import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv 



columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

df = pd.read_csv('naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

x = df.drop(columns = 'MAX_VREDNOST')
y=df['MAX_VREDNOST']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 100)

from sklearn.linear_model import LassoLarsIC
from sklearn import linear_model
reg = linear_model.LassoLarsIC(criterion='bic', normalize=False)
reg.fit(x, y)
LassoLarsIC(criterion='bic', normalize=False)
print(reg.coef_)

c = []
c.append(reg.coef_)
fields =df.drop(columns = 'MAX_VREDNOST')


filename = "RezultatJETO.csv"
rows = c
newdf=pd.DataFrame()

for i in fields:
    newdf[i] = 0
#print(newdf)

# writing to csv file 
with open(filename, 'w', newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)

df = pd.read_csv('RezultatJETO.csv', delimiter=";", decimal=",", header=0)
#print(df)


#-------!!!!!!!!!!!!!!!
#it picks the values itself, if the value is redundant its coeficient is set to 0.
