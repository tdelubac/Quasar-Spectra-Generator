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
parser.add_argument("-f", "--file", help="list of files to plot", type=str, metavar="file1 file2 ... filen", nargs='+')
arg = parser.parse_args()
files = arg.file

plt.figure()
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$F - \bar{F}$')
plt.grid(True,ls='--')
plt.title("Simulated QSO")
for file in files:
	f = ascii.read(file)
	plt.plot(f['lambda'],f['flux'],label=file)
plt.legend()
plt.show()