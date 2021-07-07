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

from pnlf.regions import Regions
from reproject import reproject_interp
from dust_extinction.parameter_averages import O94, CCM89

import matplotlib.pyplot as plt
from tqdm import tqdm

'''
logging.basicConfig(stream=sys.stdout,
                    #format='(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
'''

basedir = Path('..')
data_ext = Path('a:')

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

sample_table = ascii.read(basedir/'..'/'pnlf'/'data'/'interim'/'sample.txt')
sample_table.add_index('name')

# nebulae catalogue from Francesco (mostly HII-regions)
with fits.open(data_ext / 'MUSE_DR2.1' / 'Nebulae catalogue' / 'Nebulae_catalogue_v2.fits') as hdul:
    nebulae = Table(hdul[1].data)
nebulae['FUV_FLUX'] = np.nan
nebulae['FUV_FLUX_ERR'] = np.nan
nebulae['FUV_FLUX_CORR'] = np.nan
nebulae['FUV_FLUX_CORR_ERR'] = np.nan

astrosat_sample =set([x.stem.split('_')[0] for x in (data_ext/'Astrosat').iterdir() if x.is_file() and x.suffix=='.fits'])

for name in tqdm(sorted(astrosat_sample)):
    
    print(f'start with {name}')
    p = {x:sample_table.loc[name][x] for x in sample_table.columns}

    filename = data_ext / 'MUSE_DR2.1' / 'MUSEDAP' / f'{name}_MAPS.fits'
    with fits.open(filename) as hdul:
        Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                        uncertainty=StdDevUncertainty(hdul['HA6562_FLUX_ERR'].data),
                        mask=np.isnan(hdul['HA6562_FLUX'].data),
                        meta=hdul['HA6562_FLUX'].header,
                        wcs=WCS(hdul['HA6562_FLUX'].header))


    filename = data_ext / 'MUSE_DR2.1' / 'Nebulae catalogue' /'spatial_masks'/f'{name}_nebulae_mask.fits'
    with fits.open(filename) as hdul:
        nebulae_mask = NDData(hdul[0].data.astype(float),mask=Halpha.mask,meta=hdul[0].header,wcs=WCS(hdul[0].header))
        nebulae_mask.data[nebulae_mask.data==-1] = np.nan

    print(f'read in nebulae catalogue')
    
    # whitelight image
    astro_file = data_ext / 'Astrosat' / f'{name}_FUV_F148W_flux_reproj.fits'

    if not astro_file.is_file():
        astro_file = data_ext / 'Astrosat' / f'{name}_FUV_F154W_flux_reproj.fits'
        if not astro_file.is_file():
            print(f'no astrosat file for {name}')

    with fits.open(astro_file) as hdul:
        d = hdul[0].data
        astrosat = NDData(hdul[0].data,meta=hdul[0].header,wcs=WCS(hdul[0].header))
    print(f'read in astrosat data')
    
    muse_regions = Regions(mask=nebulae_mask.data,projection=nebulae_mask.meta,bkg=-1)
    astrosat_regions = muse_regions.reproject(astrosat.meta)
    print('regions reprojected')
    
    muse_reproj, footprint = reproject_interp((nebulae_mask.mask,nebulae_mask.wcs),astrosat.meta)
    mean,median,std=sigma_clipped_stats(astrosat.data[footprint.astype(bool)])
    print('measuring sigma_clipped_stats')
    
    tmp = nebulae[nebulae['gal_name']==name]

    flux = np.array([np.sum(astrosat.data[astrosat_regions.mask==ID]) for ID in tmp['region_ID']])
    err = np.array([np.sqrt(median**2 * len(astrosat_regions.coords[astrosat_regions.labels.index(ID)][0])) for ID in tmp['region_ID']])
    print('measuring flux')
    
    # E(B-V) is estimated from nebulae. E(B-V)_star = 0.5 E(B-V)_nebulae. FUV comes directly from stars
    extinction_mw  = extinction_model.extinguish(1481*u.angstrom,Ebv=0.5*p['E(B-V)'])
    ext_int,ext_int_err = extinction(0.5*tmp['EBV'],tmp['EBV_ERR'],wavelength=1481*u.angstrom)

    nebulae['FUV_FLUX'][nebulae['gal_name']==name] = 1e20*flux / extinction_mw
    nebulae['FUV_FLUX_ERR'][nebulae['gal_name']==name] = 1e20*err / extinction_mw

    nebulae['FUV_FLUX_CORR'][nebulae['gal_name']==name] = 1e20*flux / extinction_mw / ext_int 
    nebulae['FUV_FLUX_CORR_ERR'][nebulae['gal_name']==name] =  nebulae['FUV_FLUX_CORR'][nebulae['gal_name']==name] *np.sqrt((err/flux)**2 + (ext_int_err/ext_int)**2)  

    print('extinction correction and write to catalogue\n')
    
# write to file
primary_hdu = fits.PrimaryHDU()
table_hdu   = fits.BinTableHDU(nebulae[['gal_name','region_ID','FUV_FLUX','FUV_FLUX_ERR','FUV_FLUX_CORR','FUV_FLUX_CORR_ERR']])
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto(basedir/'data'/'interim'/'Nebulae_Catalogue_v2p1_FUV.fits',overwrite=True)