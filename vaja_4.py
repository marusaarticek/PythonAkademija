import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model

#preberemo datoteko in uredimo vrednosti izbranih atributov
df = pd.read_csv('naloga.csv', delimiter=";", header=0)
df['MAX_VREDNOST'] = df['MAX_VREDNOST'].apply(lambda x : float(x.replace(",",".")))
df['MAX_PRET_TED'] = df['MAX_PRET_TED'].apply(lambda x : float(x.replace(",",".")))

import numpy as np
import matplotlib.pyplot as plt

# W3School:
# Implementacija izracuna linearne interpolacije s pythonom

# funkcija ki kot argumente dobi x in y spremenljivki
# in izracuna regresijska koeficienta b_1 in b_0, ki predstavljata
# smerni koeficient in zacetno vrednost oz presecisce z y osjo

def estimate_coef(x, y):

	# stevilo vseh tock/vrednosti
	n = np.size(x)

	# povprecne vrednosti x in y
	m_x = np.mean(x)
	m_y = np.mean(y)

	#enacba za izracun koeficientov
	#izracunamo vsoto vseh produktov in 
	#odstejemo produkt n in povprecnih vrednosti:

	# calculating cross-deviation and deviation about x
	SS_xy = np.sum(y*x) - n*m_y*m_x
	SS_xx = np.sum(x*x) - n*m_x*m_x

	# calculating regression coefficients
	b_1 = SS_xy / SS_xx
	b_0 = m_y - b_1*m_x

	return (b_0, b_1)

#funkcija za izris grafa

def plot_regression_line(x, y, b):
	# izris vseh tock na graf
	plt.scatter(x, y, color = "m",
			marker = "o", s = 30)

	#enacba premice s podanima koeficientoma
	# predicted response vector
	y_pred = b[0] + b[1]*x

	# izris regresijske premice
	plt.plot(x, y_pred, color = "g")

	# putting labels
	plt.xlabel('x')
	plt.ylabel('y')

	# function to show plot
	plt.show()

def main():
	# podatki za x in y
	y = df['MAX_VREDNOST']# dependent variable
	x = df['MAX_PRET_TED']# independent variable

	#poklicemo funkciji
	b = estimate_coef(x, y)
	print("Estimated coefficients:\nb_0 = {} \
		\nb_1 = {}".format(b[0], b[1]))

	plot_regression_line(x, y, b)

if __name__ == "__main__":
	main()


