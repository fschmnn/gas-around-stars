'''

this site as a few infos on how to convert e/s from the HST images to physical units
https://hst-docs.stsci.edu/hstdhb/4-hst-data-analysis/4-6-analyzing-hst-images


F275W (NUV), F336W (U), F438W (B), F555W (V),
F814W (I). 


'''

from pathlib import Path
import logging
import sys

import numpy as np

from astropy.io import ascii, fits
from astropy.table import Table 
from astropy.nddata import NDData, StdDevUncertainty, InverseVariance
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import astropy.units as u 

from astrotools.regions import Regions
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

HSTbands_wave = {'NUV':2704*u.AA,'U':3355*u.AA,'B':4325*u.AA,'V':5308*u.AA,'I':8024*u.AA}
freq_to_wave = lambda band: u.mJy.to(u.erg/u.s/u.cm**2/u.Angstrom,equivalencies=u.spectral_density(HSTbands_wave[band]))

basedir = Path('..')
data_ext = Path('a:')/'Archive'

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
 
    return ext,ext_err

sample_table = ascii.read(basedir/'..'/'pnlf'/'data'/'interim'/'sample.txt')
sample_table.add_index('name')

# nebulae catalogue from Francesco (mostly HII-regions)
with fits.open(data_ext / 'Products' / 'Nebulae catalogue' / 'Nebulae_catalogue_v2.fits') as hdul:
    nebulae = Table(hdul[1].data)
nebulae['NUV_FLUX'] = np.nan
nebulae['NUV_FLUX_ERR'] = np.nan
nebulae['NUV_FLUX_CORR'] = np.nan
nebulae['NUV_FLUX_CORR_ERR'] = np.nan


for gal_name in sample_table['name']:

    print(f'start with {gal_name}')
    p = {x:sample_table.loc[gal_name][x] for x in sample_table.columns}

    filename = data_ext / 'MUSE'/ 'DR2.1' / 'MUSEDAP' / f'{gal_name}_MAPS.fits'
    with fits.open(filename) as hdul:
        Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                        uncertainty=StdDevUncertainty(hdul['HA6562_FLUX_ERR'].data),
                        mask=np.isnan(hdul['HA6562_FLUX'].data),
                        meta=hdul['HA6562_FLUX'].header,
                        wcs=WCS(hdul['HA6562_FLUX'].header))


    filename = data_ext / 'Products' / 'Nebulae catalogue' /'spatial_masks'/f'{gal_name}_nebulae_mask.fits'
    with fits.open(filename) as hdul:
        nebulae_mask = NDData(hdul[0].data.astype(float),mask=Halpha.mask,meta=hdul[0].header,wcs=WCS(hdul[0].header))
        nebulae_mask.data[nebulae_mask.data==-1] = np.nan

    print(f'read in nebulae catalogue')
    
    # NUV image
    filename = data_ext / 'HST' / 'filterImages' / f'hlsp_phangs-hst_hst_wfc3-uvis_{gal_name.lower()}_f275w_v1_exp-drc-sci.fits'
    error_file = data_ext / 'HST' / 'filterImages' / f'hlsp_phangs-hst_hst_wfc3-uvis_{gal_name.lower()}_f275w_v1_err-drc-wht.fits'

    if not filename.is_file():
        print(f'no NUV data for {gal_name}')
        continue
    else:
        with fits.open(filename) as hdul:
            F275 = NDData(hdul[0].data,
                            mask=hdul[0].data==0,
                            meta=hdul[0].header,
                            wcs=WCS(hdul[0].header))
            with fits.open(error_file) as hdul:
                F275.uncertainty = InverseVariance(hdul[0].data)
    print(f'read in HST data')
    
    muse_regions = Regions(mask=nebulae_mask.data,projection=nebulae_mask.meta,bkg=-1)
    hst_regions = muse_regions.reproject(F275.meta)
    print('regions reprojected')
    
    muse_reproj, footprint = reproject_interp((nebulae_mask.mask,nebulae_mask.wcs),F275.meta)
    mean,median,std=sigma_clipped_stats(F275.data[footprint.astype(bool)])
    print('measuring sigma_clipped_stats')
    
    tmp = nebulae[nebulae['gal_name']==gal_name]

    std_err_map = np.sqrt(1/F275.uncertainty.array)

    flux = np.array([np.sum(F275.data[hst_regions.mask==ID]) for ID in tmp['region_ID']])
    err  = np.array([np.sqrt(np.sum(std_err_map[hst_regions.mask==ID]**2)) for ID in tmp['region_ID']])

    # convert counts to physical units
    flux = np.array(flux) * F275.meta['PHOTFLAM']
    err  = np.array(err) * F275.meta['PHOTFLAM']
    print('measuring flux')
    
    # E(B-V) is estimated from nebulae. E(B-V)_star = 0.5 E(B-V)_nebulae. NUV comes directly from stars
    extinction_mw  = extinction_model.extinguish(2704*u.angstrom,Ebv=0.5*p['E(B-V)'])
    ext_int,ext_int_err = extinction(0.5*tmp['EBV'],tmp['EBV_ERR'],wavelength=2704*u.angstrom)

    nebulae['NUV_FLUX'][nebulae['gal_name']==gal_name] = 1e20*flux / extinction_mw
    nebulae['NUV_FLUX_ERR'][nebulae['gal_name']==gal_name] = 1e20*err / extinction_mw

    nebulae['NUV_FLUX_CORR'][nebulae['gal_name']==gal_name] = 1e20*flux / extinction_mw / ext_int 
    nebulae['NUV_FLUX_CORR_ERR'][nebulae['gal_name']==gal_name] =  nebulae['NUV_FLUX_CORR'][nebulae['gal_name']==gal_name] *np.sqrt((err/flux)**2 + (ext_int_err/ext_int)**2)  

    print('extinction correction and write to catalogue\n')
    
# write to file
primary_hdu = fits.PrimaryHDU()
table_hdu   = fits.BinTableHDU(nebulae[['gal_name','region_ID','NUV_FLUX','NUV_FLUX_ERR','NUV_FLUX_CORR','NUV_FLUX_CORR_ERR']])
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto('Nebulae_Catalogue_v2p1_NUV.fits',overwrite=True)