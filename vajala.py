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


# BIC: kriterij, ki pomaga pri izbiri modela. 
# When fitting models, it is possible to increase the likelihood by adding parameters,
# but doing so may result in overfitting. BIC attempts to resolve this problem by introducing
# a penalty term for the number of parameters in the model; 
# the penalty term is larger in BIC than in AIC for sample sizes greater than 7


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

x = sm.add_constant(x)

model=sm.OLS(y, x).fit()

print(model.summary())
print(model.bic)