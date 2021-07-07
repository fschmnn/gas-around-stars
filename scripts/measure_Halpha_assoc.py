from pathlib import Path
import logging
import sys

import numpy as np

from astropy.io import ascii, fits
from astropy.table import Table 
from astropy.nddata import NDData, StdDevUncertainty, Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import astropy.units as u 
from astropy.coordinates import SkyCoord

from reproject import reproject_interp
from dust_extinction.parameter_averages import O94, CCM89

import matplotlib.pyplot as plt
from tqdm import tqdm

from reproject import reproject_exact, reproject_interp
from cluster.io import read_associations
from skimage.segmentation import find_boundaries


# 'NGC0628','NGC1365','NGC1433', 'NGC1566', 'NGC3351', 'NGC3627', 'NGC4535'
name = 'NGC3627'


basedir = Path('..')
data_ext = Path('a:')

extinction_model = O94(Rv=3.1)

# we need EBV for the extinction correction
sample_table = ascii.read(basedir/'..'/'pnlf'/'data'/'interim'/'sample.txt')
sample_table.add_index('name')

def measure_dig(data,mask,label,position,factor=1,max_iter=10,size=32,plot=False):
    '''measure the diffuse ionized gas around an HII-region'''
    
    if mask.data.shape != data.shape:
        raise ValueError('data and mask have different shape')
        
    if position[0]>data.shape[0] or position[1]>data.shape[1]:
        return 6*[np.nan]

    cutout_mask = Cutout2D(mask.data,position,size=(size,size),mode='partial',fill_value=np.nan)
    cutout_data = Cutout2D(data,position,size=(size,size),mode='partial',fill_value=np.nan)
    
    area_mask  = np.sum(cutout_mask.data==label)
    input_mask = cutout_mask.data==label
    
    n_iter = 0
    while True:
        n_iter+=1
        boundaries = find_boundaries(input_mask,mode='outer')
        input_mask |=boundaries
        area_boundary = np.sum(input_mask & np.isnan(cutout_mask.data)) 
        if area_boundary > factor*area_mask or n_iter>max_iter: break
            
    if plot:
        fig,ax=plt.subplots(figsize=(5,5))
        ax.imshow(cutout_mask.data,origin='lower')
        mask = np.zeros((*cutout_mask.shape,4))
        mask[input_mask & np.isnan(cutout_mask.data),:] = (1,0,0,0.5)
        ax.imshow(mask,origin='lower')
        plt.show()
        
    #if np.sum(boundaries & np.isnan(cutout_mask.data))==0:
    #    print(f'no boundaries for {label}')
    dig = cutout_data.data[input_mask & np.isnan(cutout_mask.data)]
    hii = cutout_data.data[cutout_mask.data==label]


    return np.median(dig),np.mean(dig),np.sum(dig), np.median(hii), np.mean(hii), np.sum(hii)


print(f'reading data for {name}')
filename = data_ext / 'MUSE_DR2.1' / 'MUSEDAP' / f'{name}_MAPS.fits'
with fits.open(filename) as hdul:
    Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                    uncertainty=StdDevUncertainty(hdul['HA6562_FLUX_ERR'].data),
                    mask=np.isnan(hdul['HA6562_FLUX'].data),
                    meta=hdul['HA6562_FLUX'].header,
                    wcs=WCS(hdul['HA6562_FLUX'].header))
    Hbeta = NDData(data=hdul['HB4861_FLUX'].data,
                    uncertainty=StdDevUncertainty(hdul['HB4861_FLUX_ERR'].data),
                    mask=np.isnan(hdul['HB4861_FLUX'].data),
                    meta=hdul['HB4861_FLUX'].header,
                    wcs=WCS(hdul['HB4861_FLUX'].header))

associations, associations_mask = read_associations(folder=data_ext/'HST',target=name.lower(),scalepc=32)

tbl = associations[['assoc_ID','X','Y']].copy()
tbl['HA6562_flux'] = np.nan
tbl['HB4861_flux'] = np.nan

tbl['dig_median'] = np.nan 
tbl['dig_mean'] = np.nan 
tbl['dig_sum'] = np.nan 
tbl['hii_median'] = np.nan 
tbl['hii_mean'] = np.nan 
tbl['hii_sum'] = np.nan 

'''
if name == 'NGC3627':
    print('using cutout for NGC3627')
    associations_mask = Cutout2D(associations_mask.data,(3000,4800),size=(9000,5500),wcs=associations_mask.wcs)
    x,y = list(zip(*[associations_mask.to_cutout_position((row['X'],row['Y'])) for row in tbl]))
    tbl['X'] = x
    tbl['Y'] = y
'''


print(f'reprojecting data')
Halpha_repro = reproject_exact(Halpha,output_projection=associations_mask.wcs,
                               shape_out=associations_mask.data.shape,return_footprint=False)
Hbeta_repro = reproject_exact(Hbeta,output_projection=associations_mask.wcs,
                               shape_out=associations_mask.data.shape,return_footprint=False)


p = {x:sample_table.loc[name][x] for x in sample_table.columns}

print('measure DIG')
for row in tbl:
    row[['dig_median','dig_mean','dig_sum','hii_median','hii_mean','hii_sum']] = measure_dig(Halpha_repro,associations_mask,row['assoc_ID'],(row['X'],row['Y']))

print('measure Halpha and Hbeta')
Halpha_flux = np.array([np.sum(Halpha_repro[associations_mask.data==assoc_ID]) for assoc_ID in tbl['assoc_ID']])
Hbeta_flux = np.array([np.sum(Hbeta_repro[associations_mask.data==assoc_ID]) for assoc_ID in tbl['assoc_ID']])

# E(B-V) is estimated from nebulae. E(B-V)_star = 0.5 E(B-V)_nebulae. FUV comes directly from stars
extinction_mw  = extinction_model.extinguish(1481*u.angstrom,Ebv=0.5*p['E(B-V)'])

# divide by 25 because of pixel size?
tbl['HA6562_flux'] = Halpha_flux / extinction_mw / 25
tbl['HB4861_flux'] = Hbeta_flux / extinction_mw / 25

print('write to file')
# write to file
primary_hdu = fits.PrimaryHDU()
table_hdu   = fits.BinTableHDU(tbl)
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto(basedir/'data'/'interim'/f'{name}_associations_Halpha.fits',overwrite=True)

