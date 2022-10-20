from pathlib import Path

import numpy as np

from astropy.io import ascii, fits
from astropy.table import Table 
from astropy.nddata import NDData, StdDevUncertainty, Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import astropy.units as u 

from reproject import reproject_exact
import pyneb as pn

from skimage.segmentation import find_boundaries
from skimage.measure import moments   

import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import date

factor = 2

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

def create_annulus_mask(mask,label,factor=1,max_iter=10,plot=False):
    '''create a mask that surrounds the current HII region
    
    This function takes the mask of an HII region and creates an
    annulus mask by adding the boundary of the mask until the area
    of the annulus is larger by a factor of `factor`.
    
    Parameters
    ----------
    mask : 
        masks of the HII regions
    label :
        id of the HII region to extract
    factor :
        ratio of the area of the annulus to the HII region
    max_iter :
        number of iterations after which to stop growing the annulus mask
    plot :
        plot the HII region mask together with the new annulus mask
    
    Returns
    -------
    mask : 
        the annulus mask
    '''
    
    input_mask = mask.data==label
    area_mask  = np.sum(input_mask)
    
    # grow the mask until it reaches the desired size
    n_iter = 0
    while True:
        n_iter+=1
        boundaries = find_boundaries(input_mask,mode='outer')
        input_mask |=boundaries
        area_boundary = np.sum(input_mask & np.isnan(mask.data)) 
        if area_boundary > factor*area_mask or n_iter>max_iter: break
    
    if plot:
        M =  moments(nebulae_mask.data==label)
        y,x = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]
        cutout_mask = Cutout2D(mask.data,(x,y),size=(32,32),mode='partial',fill_value=np.nan)
        boundary_mask = Cutout2D(input_mask.data,(x,y),size=(32,32),mode='partial',fill_value=np.nan)

        fig,ax=plt.subplots(figsize=(5,5))
        ax.imshow(cutout_mask.data,origin='lower')
        mask = np.zeros((*cutout_mask.shape,4))
        mask[boundary_mask.data & np.isnan(cutout_mask.data),:] = (1,0,0,0.5)
        ax.imshow(mask,origin='lower')
        plt.show()
        
    return (input_mask & np.isnan(mask.data))


def extinction(EBV,EBV_err,wavelength,plot=False):
    '''Calculate the extinction for a given EBV and wavelength with errors
    
    Parameters
    ----------

    EBV : array

    EBV_err : array

    wavelength : float
    
    '''
    
    EBV = np.atleast_1d(EBV)
    EBV_err = np.atleast_1d(EBV_err)
    sample_size = 100000
    
    ext = pn.RedCorr(R_V=3.1,E_BV=EBV,law='CCM89 oD94').getCorr(wavelength)
    
    EBV_rand = np.random.normal(loc=EBV,scale=EBV_err,size=(sample_size,len(EBV)))
    ext_arr  = pn.RedCorr(R_V=3.1,E_BV=EBV_rand,law='CCM89 oD94').getCorr(wavelength)
    
    ext_err  = np.std(ext_arr,axis=0)
    ext_mean = np.mean(ext_arr,axis=0)
    
    if plot:
        fig,(ax1,ax2) =plt.subplots(nrows=1,ncols=2,figsize=(6,6/2))
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

nebulae['FUV_BKG_FLUX'] = np.nan
nebulae['FUV_BKG_FLUX_ERR'] = np.nan
nebulae['FUV_BKG_FLUX_CORR'] = np.nan
nebulae['FUV_BKG_FLUX_CORR_ERR'] = np.nan


astrosat_sample =set([x.stem.split('_')[0] for x in astrosat_dir.iterdir() if x.is_file() and x.suffix=='.fits'])
print(f'measuring FUV for {len(astrosat_sample)} galaxies')


astrosat_bkg = ascii.read(basedir/'data'/'external'/'astrosat_bkg.txt',delimiter='&')
astrosat_bkg.add_index('gal_name')

for gal_name in tqdm(sorted(np.unique(nebulae['gal_name']))):
    
    if gal_name not in astrosat_sample:
        print(f'no FUV data for {gal_name}')
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
    
    # the background is already extinction corrected, but the FUV maps are not
    EBV_correction = 1
    rc_MW = pn.RedCorr(R_V=3.1,E_BV=EBV_correction*EBV_MW[gal_name],law='CCM89 oD94')
    extinction_mw  = rc_MW.getCorr(1481)

    print(f'read in astrosat data')
    if gal_name in astrosat_bkg['gal_name']:
        bkg = astrosat_bkg.loc[gal_name]['bkg'] * 1e-18 / extinction_mw  
    else:
        print(f'no background for {gal_name}')
        bkg = 0
        
    astro_file = astrosat_dir / f'{gal_name}_FUV_F148W_flux_reproj.fits'
    if not astro_file.is_file():
        astro_file = astrosat_dir / f'{gal_name}_FUV_F154W_flux_reproj.fits'
        if not astro_file.is_file():
            print(f'no astrosat file for {gal_name}')

    with fits.open(astro_file) as hdul:
        d = hdul[0].data
        astrosat = NDData(hdul[0].data-bkg,meta=hdul[0].header,wcs=WCS(hdul[0].header))
        for row in hdul[0].header['COMMENT']:
            if row.startswith('CTSTOFLUX'):
                _,CTSTOFLUX = row.split(':')
                CTSTOFLUX = float(CTSTOFLUX)
            if row.startswith('IntTime'):
                _,IntTime = row.split(':')
                IntTime = float(IntTime)
    
    print('reproject regions')

    fuv_muse = reproject_exact(astrosat,nebulae_mask.meta,return_footprint=False)
    astrosat_to_muse = (astrosat.wcs.pixel_scale_matrix[0][0] / Halpha.wcs.pixel_scale_matrix[0][0])**2

    tmp = nebulae[nebulae['gal_name']==gal_name]

    print('measuring FUV flux')
    flux = np.array([np.sum(fuv_muse[nebulae_mask.data==ID]) for ID in tmp['region_ID']]) / astrosat_to_muse
    
    # this needs a loop and is slow
    bkg_flux = []
    for row in tmp:
        bkg_mask = create_annulus_mask(nebulae_mask,row['region_ID'],factor=3,max_iter=10,plot=False)
        # measure background flux and scale to area of HII region
        bkg_flux.append(np.sum(fuv_muse[bkg_mask]) / np.sum(bkg_mask) * np.sum(nebulae_mask.data==row['region_ID']) / astrosat_to_muse) 
    bkg_flux = np.array(bkg_flux)

    err  = np.sqrt(flux*CTSTOFLUX/IntTime)
    bkg_flux_err = np.sqrt(bkg_flux*CTSTOFLUX/IntTime)

    print('FUV extinction correction')
    # E(B-V) is estimated from nebulae. E(B-V)_star = 0.5 E(B-V)_nebulae. FUV comes directly from stars
    # https://ned.ipac.caltech.edu/level5/Sept12/Calzetti/Calzetti1_4.html or Calzetti+2000
    EBV_correction = 1
    rc_MW = pn.RedCorr(R_V=3.1,E_BV=EBV_correction*EBV_MW[gal_name],law='CCM89 oD94')

    extinction_mw  = rc_MW.getCorr(1481)
    ext_int,ext_int_err = extinction(EBV_correction*tmp['EBV'],tmp['EBV_ERR'],wavelength=1481)

    nebulae['FUV_FLUX'][nebulae['gal_name']==gal_name] = 1e20*flux * extinction_mw
    nebulae['FUV_FLUX_ERR'][nebulae['gal_name']==gal_name] = 1e20*err * extinction_mw

    nebulae['FUV_FLUX_CORR'][nebulae['gal_name']==gal_name] = 1e20*flux * extinction_mw * ext_int 
    nebulae['FUV_FLUX_CORR_ERR'][nebulae['gal_name']==gal_name] =  nebulae['FUV_FLUX_CORR'][nebulae['gal_name']==gal_name] *np.sqrt((err/flux)**2 + (ext_int_err/ext_int)**2)  

    nebulae['FUV_BKG_FLUX'][nebulae['gal_name']==gal_name] = 1e20*bkg_flux * extinction_mw
    nebulae['FUV_BKG_FLUX_ERR'][nebulae['gal_name']==gal_name] = 1e20*bkg_flux_err * extinction_mw

    nebulae['FUV_BKG_FLUX_CORR'][nebulae['gal_name']==gal_name] = 1e20*bkg_flux * extinction_mw * ext_int 
    nebulae['FUV_BKG_FLUX_CORR_ERR'][nebulae['gal_name']==gal_name] =  nebulae['FUV_BKG_FLUX_CORR'][nebulae['gal_name']==gal_name] *np.sqrt((bkg_flux_err/bkg_flux)**2 + (ext_int_err/ext_int)**2)  


# write to file
columns = ['gal_name','region_ID',
           'FUV_FLUX','FUV_FLUX_ERR','FUV_FLUX_CORR','FUV_FLUX_CORR_ERR',
           'FUV_BKG_FLUX','FUV_BKG_FLUX_ERR','FUV_BKG_FLUX_CORR','FUV_BKG_FLUX_CORR_ERR'
           #'HA_conv_FLUX','HA_conv_FLUX_ERR','HA_conv_FLUX_CORR','HA_conv_FLUX_CORR_ERR'
          ]
    
doc = f'''this catalogue contains the FUV fluxes for the objects in the nebula 
catalogue, measured from the Astrosat data (using the F148W filter for
all galaxies except for NGC1433 and NGC1512, for which the F154W filter
was used). All fluxes are in [f]=1e-20 erg s-1 cm-2 AA-1 and corrected 
for Milky Way foreground extinction (with the extinction curve from 
O'Donnell (1994) and E(B-V) from Schlafly & Finkbeiner (2011)). The 
columns ending with _CORR are also corrected for internal extinction, 
based on the E(B-V) from the nebula catalogue. 
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
hdul.writeto('Nebulae_Catalogue_v2p1_FUV_bkg.fits',overwrite=True)
