from pathlib import Path
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt 

from astropy.io import ascii, fits
from astropy.table import Table 
from astropy.nddata import Cutout2D, NDData
from astropy.wcs import WCS 

from skimage.segmentation import find_boundaries


basedir = Path('..')
data_ext = Path('a:')


def measure_dig(data,mask,label,position,factor=1,max_iter=10,size=32,plot=False):
    '''measure the diffuse ionized gas around an HII-region'''
    
    cutout_mask = Cutout2D(mask.data,position,size=(size,size),mode='partial',fill_value=np.nan)
    cutout_data = Cutout2D(data.data,position,size=(size,size),mode='partial',fill_value=np.nan)
    
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


# nebulae catalogue from Francesco (mostly HII-regions)
with fits.open(data_ext / 'MUSE_DR2.1' / 'Nebulae catalogue' / 'Nebulae_catalogue_v2.fits') as hdul:
    nebulae = Table(hdul[1].data)
tbl = nebulae[['gal_name','region_ID','cen_x','cen_y']].copy()
tbl['dig_median'] = np.nan 
tbl['dig_mean'] = np.nan 
tbl['dig_sum'] = np.nan 
tbl['hii_median'] = np.nan 
tbl['hii_mean'] = np.nan 
tbl['hii_sum'] = np.nan 

gal_name = None
for row in tbl:
    
    if row['gal_name']!=gal_name:
        gal_name = row['gal_name']
        print(f'start with {gal_name}')

        filename = data_ext / 'MUSE_DR2.1' / 'MUSEDAP' / f'{gal_name}_MAPS.fits'
        with fits.open(filename) as hdul:
            Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                            mask=np.isnan(hdul['HA6562_FLUX'].data),
                            meta=hdul['HA6562_FLUX'].header,
                            wcs=WCS(hdul['HA6562_FLUX'].header))
            err = hdul['HA6562_FLUX_ERR'].data

        filename = data_ext / 'MUSE_DR2.1' / 'Nebulae catalogue' /'spatial_masks'/f'{gal_name}_nebulae_mask.fits'
        with fits.open(filename) as hdul:
            nebulae_mask = NDData(hdul[0].data.astype(float),mask=Halpha.mask,meta=hdul[0].header,wcs=WCS(hdul[0].header))
            nebulae_mask.data[nebulae_mask.data==-1] = np.nan
        
        #boundaries = find_boundaries(~np.isnan(nebulae_mask.data),mode='outer')
        #not_detected = np.sum(Halpha.data[boundaries]<3*err[boundaries])/np.sum(boundaries)
        #print(f'{gal_name}: {not_detected*100:.2f} %')

    row[['dig_median','dig_mean','dig_sum','hii_median','hii_mean','hii_sum']] = measure_dig(Halpha,nebulae_mask,row['region_ID'],(row['cen_x'],row['cen_y']))


tbl['dig/hii'] = tbl['dig_median'] / tbl['hii_median']

hdu = fits.BinTableHDU(tbl[['gal_name','region_ID','dig_median','dig_mean','dig_sum','hii_median','hii_mean','hii_sum']],name='diffuse ionized gas')
hdu.writeto(basedir/'data'/'interim'/f'Nebulae_Catalogue_v2p1_dig.fits',overwrite=True)
    


