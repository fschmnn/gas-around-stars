from pathlib import Path
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt 

from astropy.io import fits
from astropy.table import Table, join

import pyneb as pn

diags = pn.Diagnostics()

# first we need to specify the path to the raw data
basedir = Path('..')
data_ext = Path('a:')/'Archive' #basedir / 'data' / 'raw' 

# nebulae catalogue from Francesco (mostly HII-regions)
with fits.open(data_ext / 'Products' / 'Nebulae_catalogs' / 'Nebulae_catalogue_v2' / 'Nebulae_catalogue_v2.fits') as hdul:
    nebulae = Table(hdul[1].data)
with fits.open(basedir/'data'/'interim'/f'Nebulae_Catalogue_v2p1_refitNII.fits') as hdul:
    refitNII = Table(hdul[1].data)
nebulae = join(nebulae,refitNII,keys=['gal_name','region_ID'])

# [SII]6717/[SII]6731 this is a tracer for the density
temp,density = diags.getCrossTemDen('[NII] 5755/6548', '[SII] 6731/6716', 
                     nebulae['NII5754_FLUX_CORR_REFIT']/nebulae['NII6583_FLUX_CORR']/0.34,
                     nebulae['SII6730_FLUX_CORR']/nebulae['SII6716_FLUX_CORR'])

nebulae['density'] = density
nebulae['temperature'] = temp

'''tmp = nebulae[:10]
NII5754 = pn.EmissionLine('N',2,5755,obsIntens=tmp['NII5754_FLUX_CORR_REFIT'],
                obsError=tmp['NII5754_FLUX_CORR_REFIT_ERR'])
NII6548 = pn.EmissionLine('N',2,6548,obsIntens=0.34*tmp['NII6583_FLUX_CORR'],
                obsError=0.34*tmp['NII6583_FLUX_CORR_ERR'])
SII6730 = pn.EmissionLine('S',2,6731,obsIntens=tmp['SII6730_FLUX_CORR'],
                obsError=tmp['SII6730_FLUX_CORR_ERR'])
SII6716 = pn.EmissionLine('S',2,6716,obsIntens=tmp['SII6716_FLUX_CORR'],
                obsError=tmp['SII6716_FLUX_CORR_ERR'])
obs = pn.Observation()
obs.addLine(NII5754)
obs.addLine(NII6548)
obs.addLine(SII6716)
obs.addLine(SII6730)

Te, Ne = diags.getCrossTemDen('[NII] 5755/6548', '[SII] 6731/6716', obs=obs)'''


def TempDen(tmp,sample_size = 1000):
    '''compute Temperature and Density with uncertainties
    
    Warning: this is super slow
    '''
    
    NII5754 = np.random.normal(loc=tmp['NII5754_FLUX_CORR'],scale=tmp['NII5754_FLUX_CORR_ERR'],size=(sample_size,len(tmp)))
    NII6583 = np.random.normal(loc=tmp['NII6583_FLUX_CORR'],scale=tmp['NII6583_FLUX_CORR_ERR'],size=(sample_size,len(tmp)))
    SII6730 = np.random.normal(loc=tmp['SII6730_FLUX_CORR'],scale=tmp['SII6730_FLUX_CORR_ERR'],size=(sample_size,len(tmp)))
    SII6716 = np.random.normal(loc=tmp['SII6716_FLUX_CORR'],scale=tmp['SII6716_FLUX_CORR_ERR'],size=(sample_size,len(tmp)))

    # calculcate for an array of values and use the std as the uncertainty
    temp,density = diags.getCrossTemDen('[NII] 5755/6548', '[SII] 6731/6716', 
                         NII5754/NII6583/0.34,SII6730/SII6716)
    Te_err = np.nanstd(temp,axis=0)
    ne_err = np.nanstd(density,axis=0)
    
    Te,ne = diags.getCrossTemDen('[NII] 5755/6548', '[SII] 6731/6716', 
                         tmp['NII5754_FLUX_CORR']/tmp['NII6583_FLUX_CORR']/0.34,
                         tmp['SII6730_FLUX_CORR']/tmp['SII6716_FLUX_CORR'])
    
    return Te,Te_err,ne,ne_err

hdu = fits.BinTableHDU(nebulae[['gal_name','region_ID','density','temperature']],name='density')
hdu.writeto(basedir/'data'/'interim'/f'Nebulae_Catalogue_v2p1_density_refit.fits',overwrite=True)