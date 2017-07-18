#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pickle

from astropy.io import fits
from astropy.io import ascii
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.stats import lognorm

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 14

''' Parsing arguments '''
parser = ArgumentParser()
parser.add_argument("-d", "--display", help="Plot spectra", dest='display', action='store_true')
parser.set_defaults(display=False)
parser.add_argument("-m", "--model", help="Model of PCA to generate the QSO", type=str, metavar="PATH_TO_MODEL", default="")
parser.add_argument("-o", "--output_file", help="Output file for the spectrum - Will overwrite existing file", type=str, metavar="OUTPUT_FILE", default="../output/spectrum.txt")
parser.add_argument("-p", "--set_p_values", help="Set the p_value at which to draw the coefficients of the components from the cumulative pdf (cdf) - no need to specify all the p_value of all the components", type=float, metavar="p_value_1 p_value_2 ... p_value_n", nargs='+')
parser.add_argument("-s", "--seed", help="Seed for random generator", type=int, metavar="SEED", default=0)
parser.add_argument("-z", "--redshift", help="Redshift of the generated spectrum", type=float, metavar="REDSHIFT")
arg = parser.parse_args()
model_file = arg.model
redshift = arg.redshift
output_file = arg.output_file
seed = arg.seed
display = arg.display
input_p_values = arg.set_p_values
np.random.seed(seed)

''' Loading model '''
with open(model_file,"rb") as model:
	pca_redshifts = np.array(pickle.load(model,encoding='latin1'))
	pcas = pickle.load(model,encoding='latin1')
	lambda_centers = pickle.load(model,encoding='latin1')
	mu = pickle.load(model,encoding='latin1')
	sigma = pickle.load(model,encoding='latin1')
	lognormal = pickle.load(model,encoding='latin1')

''' Finding model for given redshift '''
delta_z = abs(pca_redshifts - redshift)
ind = np.where(delta_z == delta_z.min())[0][0]
print('Delta_z between spectrum and model:', delta_z.min())

''' Computing coeffs '''
bins_norm = np.linspace(-100,100,20000)
bins_lognorm = np.linspace(0,1000000,10000)

p_values = np.ones(len(mu[ind])) * -99
p_values[:len(input_p_values)] = np.array(input_p_values)

coeffs = np.zeros(len(p_values))
for i,ip_value in enumerate(p_values):
	if ip_value==-99: # Draw random number from distrib
		if lognormal[ind][i]==0:
			coeffs[i] = np.random.normal(mu[ind][i],sigma[ind][i])
		else:
			coeffs[i] = np.random.lognormal(mu[ind][i],sigma[ind][i])
	else: # Estimate coefficient from number of sigmas
		if lognormal[ind][i]==0:
			cdf = lambda x: norm.cdf(x,mu[ind][i],sigma[ind][i])
			coeffs[i] = bins_norm[len(cdf(bins_norm)[cdf(bins_norm)<ip_value])]
		else:
			cdf = lambda x: lognorm.cdf(x,mu[ind][i],sigma[ind][i])
			coeffs[i] = bins_lognorm[max(len(cdf(bins_lognorm)[cdf(bins_lognorm)<=ip_value])-1,0)]


''' Saving spectra '''
lamb = lambda_centers[ind]*(1+redshift)
flux = pcas[ind].inverse_transform(coeffs)
ascii.write([lamb,flux],output_file,names=['lambda','flux'],overwrite=True)

''' Drawing random parameters '''
if display:
	plt.figure()
	plt.plot(lamb,flux)
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r'$F - \bar{F}$')
	plt.grid(True,ls='--')
	plt.title("Simulated QSO")
	plt.show()



