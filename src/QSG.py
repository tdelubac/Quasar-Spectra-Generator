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

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 14

''' Parsing arguments '''
parser = ArgumentParser()
parser.add_argument("-m", "--model",help="Model of PCA to generate the QSO",type=str,metavar="MODEL",default="")
parser.add_argument("-z", "--redshift",help="Redshift of the generated spectrum",type=float,metavar="REDSHIFT")
parser.add_argument("-o", "--output_file",help="Output file for the spectrum",type=str,metavar="OUTPUT_FILE",default="../output/spectrum.txt")
parser.add_argument("-s", "--seed",help="Seed for random generator",type=int,metavar="SEED",default=0)
parser.add_argument("-p", "--plot",help="Plot spectra", dest='plot', action='store_true')
parser.set_defaults(plot=False)
arg = parser.parse_args()
model_file = arg.model
redshift = arg.redshift
output_file = arg.output_file
seed = arg.seed
plot = arg.plot
print(plot)
np.random.seed(seed)

''' Loading model '''
model = open(model_file,"r")
pca_redshifts = np.array(pickle.load(model))
pcas = pickle.load(model)
lambda_centers = pickle.load(model)
mu = pickle.load(model)
sigma = pickle.load(model)
lognormal = pickle.load(model)

''' Finding model for given redshift '''
delta_z = abs(pca_redshifts - redshift)
ind = np.where(delta_z == delta_z.min())[0][0]
print('Delta_z between spectrum and model:', delta_z.min())

''' Drawing parameters ''' 
simu_coeffs = []
for imu,isigma,ilognormal in zip(mu[ind],sigma[ind],lognormal[ind]):
	if ilognormal==0:
		simu_coeffs.append(np.random.normal(imu,isigma))
	else:
		simu_coeffs.append(np.random.lognormal(imu,isigma))	

''' Saving spectra '''
lamb = lambda_centers[ind]*(1+redshift)
flux = pcas[ind].inverse_transform(simu_coeffs)
ascii.write([lamb,flux],output_file,names=['lambda','flux'])

''' Drawing random parameters '''
print(plot)
if plot:
	plt.figure()
	plt.plot(lamb,flux)
	plt.xlabel(r'$\lambda$')
	plt.ylabel(r'$F - \bar{F}$')
	plt.grid(True,ls='--')
	plt.title("Simulated QSO")
	plt.show()



