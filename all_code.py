import pandas as pd

df = pd.read_csv('Naloga.csv', delimiter=";", decimal=",", header=0)


### Linearna regresija

import numpy as np
import matplotlib.pyplot as plt

y = df['MAX_VREDNOST']
x = df[['MAX_PRET_TED']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) 

y_pred = regressor.predict(x_test) #predvidene vrednosti
 
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue') #regresijska premica

plt.title("MAX_VREDNOST vs MAX_PRET_TED") 
plt.xlabel("MAX_PRET_TED") 
plt.ylabel("MAX_VREDNOST") 
plt.show() 


