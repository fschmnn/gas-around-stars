import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date
from tqdm import tqdm

from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D, NDData
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u 
import astropy.constants as c

from skimage.segmentation import find_boundaries
from skimage.measure import moments              

import pyneb as pn

# area annulus / area HII region
factor = 3

# read the nebulae catalogue
with fits.open(Path('/data')/'Products'/'Nebulae_catalogs'/'Nebulae_catalogue_v2'/'Nebulae_catalogue_v2.fits') as hdul:
    catalogue = Table(hdul[1].data)
tmp = catalogue['gal_name','region_ID','region_area'].copy()

    
    
def extract_spectrum_with_background(data,mask,label,factor=1,max_iter=10,plot=False):
    '''extract the spectra of HII region and annulus
    
    This function takes the mask of an HII region and creates an
    annulus mask by adding the boundary of the mask until the area
    of the annulus is larger by a factor of `factor`.
    
    Parameters
    ----------
    data : 
        the spectral cube 
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
    hii_region : 
        spectrum of the HII region
    background : 
        spectrum of the annulus surrounding the HII region
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
    
    # sum up the spectra in the previously defined masks
    hii_region = np.nansum(data[...,mask.data==label],axis=1)

    # we use the median along the spatial axis
    continuum  = np.nanmedian(data[...,input_mask & np.isnan(mask.data)],axis=1)
    area_continuum = np.nansum(input_mask & np.isnan(mask.data))
    
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
        
    # we need to scale the continuum to the size of the HII region
    return hii_region, continuum, area_continuum

spectra  = []
annuli   = []
wlam_lst = []

tmp['annulus_area'] = np.nan
    
# the main loop
gal_name = None
for row in tqdm(tmp):
    # if the galaxy changes we need to read in the data for the new galaxy
    if gal_name!=row['gal_name']:
        gal_name=row['gal_name']
        print(f'reading {gal_name}')
        # read the nebulae mask
        filename = Path('/data')/'Products'/'Nebulae_catalogs'/'Nebulae_catalogue_v2'/'spatial_masks'/f'{gal_name}_nebulae_mask_V2.fits'
        with fits.open(filename) as hdul:
            nebulae_mask = NDData(hdul[0].data.astype(float),meta=hdul[0].header,wcs=WCS(hdul[0].header))
            nebulae_mask.data[nebulae_mask.data==-1] = np.nan
        # read the data cube
        filename = Path('/data')/'MUSE'/'DR2.1'/'native'/'datacubes'/f'{gal_name}_DATACUBE_FINAL_WCS_Pall_mad.fits'
        with fits.open(filename , memmap=False, mode='denywrite') as hdul:
            #cube=SpectralCube(data=hdul[1].data,wcs=WCS(hdul[1].header))
            data_cube   = hdul[1].data
            cube_header = hdul[1].header
        # this is the same for all nebulae
        wlam = np.linspace(cube_header['CRVAL3'],cube_header['CRVAL3']+cube_header['NAXIS3']*cube_header['CD3_3'],cube_header['NAXIS3'])
        # we also need to update the extinction correction

    # determine the redshift of the current nebulae

    spec,bkg, area_bkg = extract_spectrum_with_background(data_cube,nebulae_mask,row['region_ID'],factor=factor)    

    spectra.append(spec)
    annuli.append(bkg)
    wlam_lst.append(wlam)

    row['annulus_area'] = area_bkg

tmp['wlam'] = wlam_lst
tmp['region_spectra'] = spectra
tmp['annulus_spectra'] = annuli

doc = f'''Extract annuli spectra
Based on `Nebulae_catalogue_v2.fits`. 
last update: {date.today().strftime("%b %d, %Y")}
'''

primary_hdu = fits.PrimaryHDU()
for i,comment in enumerate(doc.split('\n')):
    if i==0:
        primary_hdu.header['COMMENT'] = comment
    else:
        primary_hdu.header[''] = comment
table_hdu   = fits.BinTableHDU(tmp)
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto('Nebulae_Catalogue_v2p1_annuli_spectra.fits',overwrite=True)
    