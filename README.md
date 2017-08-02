# Quasar Spectra Generator 

The Quasar Spectra Generator (QSG) is a software intended to generates quasar spectra. Given a set of quasar spectra, it performs a principal component analysis (PCA) to decompose the original set, and is then able to generate random quasar templates from the principal components. It is made of two principal pieces of code: 

1. QSG_fit.py that performs the PCA 
2. QSG.py that generates spectra given a model obtained by QSG_fit.py

The code comes with a precomputed model based on <a href="http://www.sdss.org/dr12/">SDSS-III BOSS DR12 data</a> able to generate quasar spectra from redshift 0 to 4 per bin of 0.1. 

## QSG_fit

QSG_fit.py performes the PCA, producing a file containing a model that can go as an input of QSG.py. The code should be provided with a directory containing only the input spectra in table fits format. There should be one file per bin in redshift containing as many spectra as desired. The table should at minimum contain the following columns:
* Z_VI       redshift of the spectrum
* LAMBDA     wavelength in observer frame
* FLUX       flux in arbitrary units
* IVAR_FLUX  inverse variance of the flux 

For a list of available commandes type:
'''
./QSG_fit.py --help
'''
from the src directory.

## QSG

QSG.py generates random quasar templates given a model computed with QSG_fit.py.

For a list of available commandes type:
'''
./QSG.py --help
'''
from the src directory.



