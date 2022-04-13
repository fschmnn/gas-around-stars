from pathlib import Path
import logging
import sys

import numpy as np

from astropy.io import ascii, fits
from astropy.table import Table 
from astropy.nddata import NDData, StdDevUncertainty
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import astropy.units as u 

from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft

import astrotools as tools
from astrotools.regions import Regions
from reproject import reproject_interp
from dust_extinction.parameter_averages import O94, CCM89

import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import date

basedir = Path('..')
data_ext = Path('a:')/'Archive'

astrosat_dir = data_ext / 'Astrosat'
nebulae_dir  = data_ext / 'Products' / 'Nebulae_catalogs' / 'Nebulae_catalogue_v2'
muse_dir     = data_ext / 'MUSE' / 'DR2.1' / 'copt' / 'MUSEDAP'

# Milky Way E(B-V) from  Schlafly & Finkbeiner (2011)
EBV_MW = {'IC5332': 0.015,'NGC0628': 0.062,'NGC1087': 0.03,'NGC1300': 0.026,
          'NGC1365': 0.018,'NGC1385': 0.018,'NGC1433': 0.008,'NGC1512': 0.009,
          'NGC1566': 0.008,'NGC1672': 0.021,'NGC2835': 0.089,'NGC3351': 0.024,
          'NGC3627': 0.037,'NGC4254': 0.035,'NGC4303': 0.02,'NGC4321': 0.023,
          'NGC4535': 0.017,'NGC5068': 0.091,'NGC7496': 0.008}

extinction_model = O94(Rv=3.1)

def extinction(EBV,EBV_err,wavelength,plot=False):
    '''Calculate the extinction for a given EBV and wavelength with errors'''
    
    EBV = np.atleast_1d(EBV)
    sample_size = 100000

    ext = extinction_model.extinguish(wavelength,Ebv=EBV)
    
    EBV_rand = np.random.normal(loc=EBV,scale=EBV_err,size=(sample_size,len(EBV)))
    ext_arr  = extinction_model.extinguish(wavelength,Ebv=EBV_rand)
        
    ext_err  = np.std(ext_arr,axis=0)
    ext_mean = np.mean(ext_arr,axis=0)
    
    if plot:
        fig,(ax1,ax2) =plt.subplots(nrows=1,ncols=2,figsize=(two_column,two_column/2))
        ax1.hist(EBV_rand[:,0],bins=100)
        ax1.axvline(EBV[0],color='black')
        ax1.set(xlabel='E(B-V)')
        ax2.hist(ext_arr[:,0],bins=100)
        ax2.axvline(ext[0],color='black')
        ax2.set(xlabel='extinction')
        plt.show()
 
    return ext,ext_err

# nebulae catalogue from Francesco (mostly HII-regions)
with fits.open(nebulae_dir / 'Nebulae_catalogue_v2.fits') as hdul:
    nebulae = Table(hdul[1].data)
nebulae['FUV_FLUX'] = np.nan
nebulae['FUV_FLUX_ERR'] = np.nan
nebulae['FUV_FLUX_CORR'] = np.nan
nebulae['FUV_FLUX_CORR_ERR'] = np.nan

nebulae['HA_conv_FLUX'] = np.nan
nebulae['HA_conv_FLUX_ERR'] = np.nan
nebulae['HA_conv_FLUX_CORR'] = np.nan
nebulae['HA_conv_FLUX_CORR_ERR'] = np.nan

astrosat_sample =set([x.stem.split('_')[0] for x in astrosat_dir.iterdir() if x.is_file() and x.suffix=='.fits'])
print(f'measuring FUV for {len(astrosat_sample)} galaxies')

for gal_name in tqdm(sorted(np.unique(nebulae['gal_name']))):
    
    if gal_name not in astrosat_sample:
        continue
        
    print(f'start with {gal_name}')

    print(f'read in nebulae catalogue')
    filename = next(muse_dir.glob(f'{gal_name}*.fits'))
    copt_res = float(filename.stem.split('-')[1].split('asec')[0])
    with fits.open(filename) as hdul:
        Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                        uncertainty=StdDevUncertainty(hdul['HA6562_FLUX_ERR'].data),
                        mask=np.isnan(hdul['HA6562_FLUX'].data),
                        meta=hdul['HA6562_FLUX'].header,
                        wcs=WCS(hdul['HA6562_FLUX'].header))
    
    filename = nebulae_dir /'spatial_masks'/f'{gal_name}_nebulae_mask.fits'
    with fits.open(filename) as hdul:
        nebulae_mask = NDData(hdul[0].data.astype(float),mask=Halpha.mask,meta=hdul[0].header,wcs=WCS(hdul[0].header))
        nebulae_mask.data[nebulae_mask.data==-1] = np.nan
    
    print(f'read in astrosat data')
    astro_file = astrosat_dir / f'{gal_name}_FUV_F148W_flux_reproj.fits'
    if not astro_file.is_file():
        astro_file = astrosat_dir / f'{gal_name}_FUV_F154W_flux_reproj.fits'
        if not astro_file.is_file():
            print(f'no astrosat file for {gal_name}')

    with fits.open(astro_file) as hdul:
        d = hdul[0].data
        astrosat = NDData(hdul[0].data,meta=hdul[0].header,wcs=WCS(hdul[0].header))
        for row in hdul[0].header['COMMENT']:
            if row.startswith('CTSTOFLUX'):
                _,CTSTOFLUX = row.split(':')
                CTSTOFLUX = float(CTSTOFLUX)
            if row.startswith('IntTime'):
                _,IntTime = row.split(':')
                IntTime = float(IntTime)
        
        
    print('reproject regions')
    muse_regions = Regions(mask=nebulae_mask.data,projection=nebulae_mask.meta,bkg=-1)
    astrosat_regions = muse_regions.reproject(astrosat.meta)
    
    tmp = nebulae[nebulae['gal_name']==gal_name]

    print('measuring FUV flux')
    flux = np.array([np.sum(astrosat.data[astrosat_regions.mask==ID]) for ID in tmp['region_ID']])
    err  = np.sqrt(flux*CTSTOFLUX/IntTime)
    
    print('FUV extinction correction')
    # E(B-V) is estimated from nebulae. E(B-V)_star = 0.5 E(B-V)_nebulae. FUV comes directly from stars
    # https://ned.ipac.caltech.edu/level5/Sept12/Calzetti/Calzetti1_4.html or Calzetti+2000
    extinction_mw  = extinction_model.extinguish(1481*u.angstrom,Ebv=0.44*EBV_MW[gal_name])
    ext_int,ext_int_err = extinction(0.44*tmp['EBV'],tmp['EBV_ERR'],wavelength=1481*u.angstrom)

    nebulae['FUV_FLUX'][nebulae['gal_name']==gal_name] = 1e20*flux / extinction_mw
    nebulae['FUV_FLUX_ERR'][nebulae['gal_name']==gal_name] = 1e20*err / extinction_mw

    nebulae['FUV_FLUX_CORR'][nebulae['gal_name']==gal_name] = 1e20*flux / extinction_mw / ext_int 
    nebulae['FUV_FLUX_CORR_ERR'][nebulae['gal_name']==gal_name] =  nebulae['FUV_FLUX_CORR'][nebulae['gal_name']==gal_name] *np.sqrt((err/flux)**2 + (ext_int_err/ext_int)**2)  

    
    print('convolve Halpha')
    # we measure Halpha from a convolved image  
    
    # convolve uncertainty
    # https://iopscience.iop.org/article/10.3847/2515-5172/abe8df
    # var_conv = var x kernel^2

    # times 5 to convert from arcesc to pixel
    stddev = 5*np.sqrt(1.8**2-copt_res**2)

    kernel = Gaussian2DKernel(x_stddev=stddev)
    Halpha_conv = convolve_fft(Halpha,kernel,preserve_nan=True)
    Halpha_err_conv = np.sqrt(convolve_fft(Halpha.uncertainty.array**2,kernel._array**2,normalize_kernel=False,preserve_nan=True))

    print('measuring Halpha flux')
    flux = np.array([np.sum(Halpha_conv[muse_regions.mask==ID]) for ID in tmp['region_ID']])
    err  = np.array([np.sqrt(np.sum(Halpha_err_conv[muse_regions.mask==ID]**2)) for ID in tmp['region_ID']]) 
    
    print('Halpha extinction correction')
    extinction_mw  = extinction_model.extinguish(6562*u.angstrom,Ebv=EBV_MW[gal_name])
    ext_int,ext_int_err = extinction(tmp['EBV'],tmp['EBV_ERR'],wavelength=6562*u.angstrom)

    nebulae['HA_conv_FLUX'][nebulae['gal_name']==gal_name] = 1e20*flux / extinction_mw
    nebulae['HA_conv_FLUX_ERR'][nebulae['gal_name']==gal_name] = 1e20*err / extinction_mw

    nebulae['HA_conv_FLUX_CORR'][nebulae['gal_name']==gal_name] = 1e20*flux / extinction_mw / ext_int 
    nebulae['HA_conv_FLUX_CORR_ERR'][nebulae['gal_name']==gal_name] =  nebulae['HA_conv_FLUX'][nebulae['gal_name']==gal_name] *np.sqrt((err/flux)**2 + (ext_int_err/ext_int)**2)  

# write to file
columns = ['gal_name','region_ID',
           'FUV_FLUX','FUV_FLUX_ERR','FUV_FLUX_CORR','FUV_FLUX_CORR_ERR',
           'HA_conv_FLUX','HA_conv_FLUX_ERR','HA_conv_FLUX_CORR','HA_conv_FLUX_CORR_ERR']
    
doc = f'''this catalogue contains the FUV fluxes for the objects in the nebula 
catalogue, measured from the Astrosat data (using the F148W filter for
all galaxies except for NGC1433 and NGC1512, for which the F154W filter
was used). All fluxes are in [f]=1e-20 erg s-1 cm-2 AA-1 and corrected 
for Milky Way foreground extinction (with the extinction curve from 
O'Donnell (1994) and E(B-V) from Schlafly & Finkbeiner (2011)). The 
columns ending with _CORR are also corrected for internal extinction, 
based on the E(B-V) from the nebula catalogue. The Halpha fluxes in the 
catalogue are measured from the DAP linemap convolved to the same 
resolution as the Astrosat data (1.8"). 
Based on the nebula catalogue v2p0. 
This catalogue was created with the following script:
https://github.com/fschmnn/cluster/blob/master/scripts/measure_FUV.py
last update: {date.today().strftime("%b %d, %Y")}
'''

primary_hdu = fits.PrimaryHDU()
for i,comment in enumerate(doc.split('\n')):
    if i==0:
        primary_hdu.header['COMMENT'] = comment
    else:
        primary_hdu.header[''] = comment
table_hdu   = fits.BinTableHDU(nebulae[columns])
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto('Nebulae_Catalogue_v2p1_FUV.fits',overwrite=True)
