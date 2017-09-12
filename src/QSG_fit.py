#!/usr/bin/env python  
import numpy as np
import matplotlib.pyplot as plt
import pickle

from astropy.io import fits
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.stats import lognorm

np.random.seed(123)

''' Parsing arguments '''
parser = ArgumentParser()
parser.add_argument("-d", "--input_directory",help="directory to scan for input files",type=str,metavar="INPUT_DIRECTORY",default="../data/")
parser.add_argument("-nspec", "--number_of_spectra",help="Number of spectra to use to compute the PCA",type=int,metavar="NUMBER_OF_SPECTRA",default=100)
parser.add_argument("-ncomp", "--number_of_components",help="Number of components of the PCA",type=int,metavar="NUMBER_OF_COMPONENTS",default=10)
parser.add_argument("-o", "--output_file",help="Output file to save PCA",type=str,metavar="OUTPUT_FILE",default="../models/PCA_model.pickle")
arg = parser.parse_args()
input_directory = arg.input_directory
number_of_spectra = arg.number_of_spectra
number_of_components = arg.number_of_components
output_file = arg.output_file

''' Loading input files '''
files = [f for f in listdir(input_directory) if isfile(join(input_directory, f))]
print("Loading files:")
for f in files:
	print(f)
h = [fits.open(join(input_directory,f)) for f in files]
qso = [ih[1].data[:number_of_spectra] for ih in h] # keeping only number_of_spectra
mean_redshift = [np.mean(iqso['Z_VI']) for iqso in qso]
[ih.close() for ih in h]

''' Keeping only QSOs with mean_flux > 0'''
for iqso in qso:
        iqso = iqso[np.mean(iqso['FLUX'])>0]

''' Binning in lambda '''
lambda_min = []
lambda_max = []
for iqso in qso:
	lambda_min.append(np.max([np.min(el) for el in iqso['LAMBDA']]))
	lambda_max.append(np.min([np.max(el) for el in iqso['LAMBDA']]))
lambda_bins = []
lambda_centers = []
for ilambda_min, ilambda_max, iqso in zip(lambda_min,lambda_max,qso):
	lambda_bins.append(np.linspace(ilambda_min,ilambda_max,np.shape(iqso['LAMBDA'])[1]/20))
	lambda_centers.append((lambda_bins[-1][:-1]+lambda_bins[-1][1:])/2.)

''' This is the expansive step '''
ivars = []
specs = []
for ilambda_bins,iqso in zip(lambda_bins,qso):
	spec = []
	ivar = []
	for i in np.arange(len(iqso)):
		digitized = np.digitize(iqso[i]['LAMBDA'],ilambda_bins)
		ispec = [(iqso[i]['FLUX'][digitized == el]*iqso[i]['IVAR_FLUX'][digitized == el]).mean()/iqso[i]['IVAR_FLUX'][digitized == el].mean() if iqso[i]['IVAR_FLUX'][digitized == el].mean()>0 else 0 for el in range(1, len(ilambda_bins))]
		iivar = [iqso[i]['IVAR_FLUX'][digitized == el].mean() for el in range(1, len(ilambda_bins))]
		spec.append(ispec)
		ivar.append(iivar)
	specs.append(spec)
	ivars.append(ivar)

''' Normalise to 1 '''
for i,ispecs in enumerate(specs):
	for ii,iispecs in enumerate(ispecs):
		iispecs = np.array(iispecs)
		if iispecs.mean()>0: 
			specs[i][ii] = iispecs/iispecs.mean()
		else:
			print("warning spec "+str(ii)+" of redshift bin "+str(i)+" as mean value <= 0")

# ''' Subtracting mean '''
# for i,ispecs in enumerate(specs):
# 	for ii,iispecs in enumerate(ispecs):
# 		iispecs = np.array(iispecs)
# 		specs[i][ii] = iispecs-iispecs.mean()


''' fitting PCA '''
pcas = []
for ispecs in specs:
	X = ispecs
	pca = PCA(number_of_components)
	pca.fit(X)
	pcas.append(pca)

''' Cumulative variance explained '''
for f,ipca in zip(files,pcas):
	cumul = []
	for i in range(len(ipca.explained_variance_ratio_)):
		cumul.append(np.sum(ipca.explained_variance_ratio_[:i+1]))
	print(f,'cumulative variance explained',cumul[-1])

''' Fitting coefficients '''
mu = []
sigma = []
lognormal = []

for num,ipca in enumerate(pcas):
	imu = []
	isigma = []
	ilognormal = []

	coeffs = ipca.transform(specs[num])
	coeffs = np.transpose(coeffs)
	for i,icoeff in enumerate(coeffs):
		icoeff = np.array(icoeff)
		icoeff = icoeff[np.abs(icoeff) < 1*icoeff.std()]
		plt.figure()
		n,bins,patches = plt.hist(icoeff,bins=60,normed=1)
		if i==0:
			shape,loc,scale = lognorm.fit(icoeff)
			plt.plot(bins,lognorm.pdf(bins,shape,loc,scale))
			imu.append(np.log(scale))
			isigma.append(shape)
			ilognormal.append(True)
			plt.xlabel('Coeff')
			plt.ylabel('normalized # count')
			plt.title('PCA component '+str(i))
			plt.savefig('../sanity_plots/PCA_'+str(num)+'_component'+str(i)+'_coeffs.pdf')
			plt.close()
		else:
			temp_mu,temp_sigma = norm.fit(icoeff)
			plt.plot(bins,norm.pdf(bins,temp_mu,temp_sigma))
			imu.append(temp_mu)
			isigma.append(temp_sigma)
			ilognormal.append(False)
			plt.xlabel('Coeff')
			plt.ylabel('normalized # count')
			plt.title('PCA component '+str(i))
			plt.savefig('../sanity_plots/PCA_'+str(num)+'_component'+str(i)+'_coeffs.pdf')
			plt.close()
	mu.append(imu)
	sigma.append(isigma)
	lognormal.append(ilognormal)

''' Saving model '''
output = open(output_file, 'wb')
pickle.dump(mean_redshift, output)
pickle.dump(pcas, output)
pickle.dump(lambda_centers, output)
pickle.dump(mu,output)
pickle.dump(sigma,output)
pickle.dump(lognormal,output)
output.close()


