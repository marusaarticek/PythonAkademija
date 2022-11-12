"""
Naloga 7

Mislim, da nisem dobro razumela naloge ali pa sem si jo razložila
napačno.
Skušala sem najti rešitev za izračun najmanjše vrednosti BIC
za vse možne kombinacije atributov, a se takoj pojavi problem, saj
je računanje vseh kombinacij časovno zahtevno.

Koda za izracun:
"""
import itertools
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

plt.style.use('ggplot')


columns_to_skip = ['ID', 'DAN', 'LETO', 'DAN_V_TEDNU', 'TEDEN', 'DAN_V_MESECU', 'DATUM', 'STEVEC', 'MIN_VREDNOST']

#Izbrani podatki
df = pd.read_csv('naloga.csv', delimiter=";", decimal=",", header=0, usecols=lambda x: x not in columns_to_skip)

# Izberi podatke za x in y
X = df.drop(columns = 'MAX_VREDNOST')
Y=df['MAX_VREDNOST']

def fit_linear_reg(X,Y):
    #Fit linear regression model and return BIC and R squared values
   
    X = sm.add_constant(X)
    model_k = sm.OLS(Y, X).fit()

    #tu implementas za BIC
    BIC = model_k.bic
    return BIC

from tqdm import tnrange, tqdm_notebook
 
BIC_list, feature_list = [],[]
numb_features = []

#Looping over k = 1 to k = 11 features in X
for k in range(1, 2):

    #Looping over all possible combinations: from 11 choose k
    for combo in itertools.combinations(X.columns,k):
        tmp_result = fit_linear_reg(X[list(combo)],Y)   #Store temp result 
        BIC_list.append(tmp_result)                  #Append lists
        feature_list.append(combo)
        numb_features.append(len(combo))   

#Store in DataFrame
df = pd.DataFrame({'BIC': BIC_list,'numb_features': numb_features,'features':feature_list})

#df_min = df[df.groupby('numb_features')['BIC'].transform(min) == df['BIC']]
print(df.head(19))

"""

               BIC  ...                                           features
16     9203.762902  ...                                   (PMAX_VRED_LM1,)
34     9055.970137  ...                           (PRA_DAN, PMAX_VRED_LM1)
325    9000.149281  ...                 (PRA_DAN, TEMP_MAX, PMAX_VRED_LM1)
1593   8988.318002  ...           (PRA_DAN, BDP1, TEMP_MAX, PMAX_VRED_LM1)
7020   8977.753740  ...  (PRA_DAN, BDP1, MAX_PRET_TED, TEMP_MAX, PMAX_V...
17968  8967.711773  ...  (PRA_DAN, PRA_PRED, BDP1, MAX_PRET_TED, TEMP_M...

Rezultat pri vrednosti k=6? Razvidno je da BIC počasi pada pri dodajanju
izbranih atributov

Problem: izracunat moramo BIC za najbolso kombinacijo atributov-katere
spremenljivke najbolj vplivajo na točnost izracuna MAX VREDNOST?
Glede na to izdelamo najboljši model. Kako najdemo najboljši model? Pogledamo
več kriterijev točnosti:
1. R2, squared correlation between the observed outcome values and the predicted
 values by the model. The higher, the better.
2. RMSE, average error performed by the model-square root of MSE(mean square error)
 The lower, the better.
3.RSE, variant of RMSE adjusted for the number of predictors in the model.The loweer the better.
4. MAE, the prediction error, average absolute difference between observed and predicted outcomes.

Problem zgornjih kriterijev je, da so občutljive na dodajanje novih vrednosti v model,
tudi če te ne doprinesejo k napovedi modela.
Obstajajo še 4 pomembni kriteriji za evaluacijo in izbiro boljšega modela.
1. AIC
2. AICc
3.BIC
4.Mallows Cp

Tezava: -najti najboljso kombinacijo je casovno zahtevno.
        -vrednosti BIC za napoved MAx_vrednosti posameznih atributov
         so VSE zelo visoke in imajo majhna odstopanja
"""