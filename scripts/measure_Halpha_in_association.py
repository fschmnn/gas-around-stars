'''measure Halpha/Hbeta and E(B-V) from MUSE in the association masks

1. Read the Halpha and Hbeta linemap from the MUSEDAP
2. Read the association catalogue and mask
3. To reproject tha masks:
    1) Use skimage to find the contours of each region
    2) convert them to a astropy pixel region and convert this object to a sky region
    3) The sky region is reprojected to the MUSE data
    4) Convert the region to a mask 
    5) Measure the flux by summing the product of the mask and the linemap
4. The fluxes are corrected for Milkyway extinction
5. The Internal reddening is estimated from the Balmer decrement
6. The fluxes are corrected for internal extinction (_FLUX_CORR)
7. The catalogue is saved to a file
'''

from pathlib import Path      # handle paths to files
import sys                    # 
from tqdm import tqdm         # progress bar for long loops

import numpy as np                # arrays and useful math functions
import matplotlib.pyplot as plt   # plot stuff

from astropy.io import ascii, fits     # read/write from/to files
from astropy.table import Table        # useful data structure
from astropy.nddata import NDData, StdDevUncertainty, Cutout2D # useful when working with data
from astropy.wcs import WCS                 # astronomical coordinates systems
from astropy.coordinates import SkyCoord    # same
import astropy.units as u              # convert between different units

import pyneb    # emission line stuff (used for extinction correction)
from reproject import reproject_exact, reproject_interp     # reproject images
from skimage.measure import find_contours                   # find contours of mask
from regions import PixCoord, PolygonPixelRegion            # handle the regions

# choose which model to use
gal_name = 'IC5332'
version  = 'v1p2'
HSTband  = 'nuv'
scalepc  = 32

# the folder with the data (structure similar to Google Drive)
data_ext = Path('a:')/'Archive'

# Milky Way E(B-V) from  Schlafly & Finkbeiner (2011)
EBV_MW = {'IC5332': 0.015,'NGC0628': 0.062,'NGC1087': 0.03,'NGC1300': 0.026,
          'NGC1365': 0.018,'NGC1385': 0.018,'NGC1433': 0.008,'NGC1512': 0.009,
          'NGC1566': 0.008,'NGC1672': 0.021,'NGC2835': 0.089,'NGC3351': 0.024,
          'NGC3627': 0.037,'NGC4254': 0.035,'NGC4303': 0.02,'NGC4321': 0.023,
          'NGC4535': 0.017,'NGC5068': 0.091,'NGC7496': 0.008}


def region_from_mask(mask):
    '''take a bool mask as input and return outline of regions as PixelRegion'''

    # otherwiese we have problems with the edge of the image
    mask[:,0] = False
    mask[:,-1] = False
    mask[0,:] = False
    mask[-1,:] = False
    
    contours = find_contours(mask.astype(float),level=0.5)
    
    regs = []
    for contour in contours:
        regs.append(PolygonPixelRegion(PixCoord(*contour.T[::-1])))
     
    return regs
    #return reduce(lambda x,y:x&y,regs)


print(f'reading MUSE data for {gal_name}')
filename = data_ext / 'MUSE'/'DR2.1'/'MUSEDAP'/f'{gal_name}_MAPS.fits'
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


print(f'reading HST data for {gal_name}')
folder=data_ext/'Products'/'stellar_associations'/f'associations_{version}'/f'{gal_name.lower()}_{HSTband}'/f'{scalepc}pc'
# first the association catalogue
catalogue_file = folder / f'{gal_name.lower()}_phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits'
with fits.open(catalogue_file) as hdul:
    associations = Table(hdul[1].data)

# modify table (rename the columns such that the clusters and associations are identical)
associations['SkyCoord'] = SkyCoord(associations['reg_ra']*u.degree,associations['reg_dec']*u.degree)
associations.rename_column('reg_id','assoc_ID')

mask_file = folder / f'{gal_name.lower()}_phangshst_associations_{HSTband}_ws{scalepc}pc_idmask_{version}.fits'
with fits.open(mask_file) as hdul:
    mask = hdul[0].data.astype(float)
    mask[mask==0] = np.nan
    associations_mask = NDData(mask,
                            mask=mask==0,
                            meta=hdul[0].header,
                            wcs=WCS(hdul[0].header))


# the output table with columns for the Halpha/Hbeta fluxes
tbl = associations[['assoc_ID','reg_x','reg_y']].copy()
tbl.add_column(gal_name,index=0,name='gal_name')
tbl['HA6562_FLUX'] = np.nan
tbl['HB4861_FLUX'] = np.nan
tbl['HA6562_FLUX_ERR'] = np.nan
tbl['HB4861_FLUX_ERR'] = np.nan


'''
if name == 'NGC3627':
    print('using cutout for NGC3627')
    associations_mask = Cutout2D(associations_mask.data,(3000,4800),size=(9000,5500),wcs=associations_mask.wcs)
    x,y = list(zip(*[associations_mask.to_cutout_position((row['X'],row['Y'])) for row in tbl]))
    tbl['X'] = x
    tbl['Y'] = y

# this gives very similar results
print(f'reprojecting data')
Halpha_repro = reproject_exact(Halpha,output_projection=associations_mask.wcs,
                               shape_out=associations_mask.data.shape,return_footprint=False)
Hbeta_repro = reproject_exact(Hbeta,output_projection=associations_mask.wcs,
                               shape_out=associations_mask.data.shape,return_footprint=False)

#print('measure Halpha and Hbeta from mask')
# divide by 25 because of pixel size?
Halpha_flux = np.array([np.sum(Halpha_repro[associations_mask.data==assoc_ID]) for assoc_ID in tbl['assoc_ID']]) / 25
Hbeta_flux = np.array([np.sum(Hbeta_repro[associations_mask.data==assoc_ID]) for assoc_ID in tbl['assoc_ID']]) / 25
'''

# create new columns to save the measured fluxes with errors
tbl['HA6562_FLUX'] = np.nan
tbl['HB4861_FLUX'] = np.nan
tbl['HA6562_FLUX_ERR'] = np.nan
tbl['HB4861_FLUX_ERR'] = np.nan

for row in tqdm(tbl):

    # it is way faster to search for the regions in a cutout
    assoc_ID = row['assoc_ID']
    position = associations[associations['assoc_ID']==assoc_ID]['SkyCoord']
    
    try:
        # sometimes the associations are outside of the MUSE FOV
        cutout_Halpha=Cutout2D(Halpha.data,position,size=20*u.arcsec,wcs=Halpha.wcs)
        cutout_Hbeta=Cutout2D(Hbeta.data,position,size=20*u.arcsec,wcs=Hbeta.wcs)
        cutout_Halpha_err=Cutout2D(Halpha.uncertainty.array,position,size=20*u.arcsec,wcs=Halpha.wcs)
        cutout_Hbeta_err=Cutout2D(Hbeta.uncertainty.array,position,size=20*u.arcsec,wcs=Hbeta.wcs)
    except:
        continue

    cutout_mask=Cutout2D(associations_mask.data,position,size=20*u.arcsec,wcs=associations_mask.wcs)

    # in rare cases this will return multiple regions. we are lazy and just use the first one
    reg_pix_hst = region_from_mask(np.isin(cutout_mask.data,assoc_ID))[0]
    reg_sky = reg_pix_hst.to_sky(cutout_mask.wcs)
    reg_pix_muse = reg_sky.to_pixel(cutout_Halpha.wcs)
    mask = reg_pix_muse.to_mask(mode='subpixels',subpixels=16)
    

    try:
        # sum up the flux inside of this mask
        row['HA6562_FLUX'] = np.sum(mask.multiply(cutout_Halpha.data))
        row['HB4861_FLUX'] = np.sum(mask.multiply(cutout_Hbeta.data))
        
        # for the uncertainty we take the square root of the sum of the squared uncertainties
        row['HA6562_FLUX_ERR'] = np.sqrt(np.sum(mask.multiply(cutout_Halpha.data)**2))
        row['HB4861_FLUX_ERR'] = np.sqrt(np.sum(mask.multiply(cutout_Hbeta.data)**2))
    except:
        continue

print('correct for extinction')
# Milky Way extinction
rc_MW = pyneb.RedCorr(E_BV = EBV_MW[gal_name], R_V = 3.1, law = 'CCM89 oD94')

tbl['HA6562_FLUX'] *= rc_MW.getCorr(6562) 
tbl['HB4861_FLUX'] *= rc_MW.getCorr(4861)
tbl['HA6562_FLUX_ERR'] *= rc_MW.getCorr(6562) 
tbl['HB4861_FLUX_ERR'] *= rc_MW.getCorr(4861)

# Internal extinction is estimated from the Balmer decrement
rc_balmer = pyneb.RedCorr(R_V = 3.1, law = 'CCM89 oD94')
rc_balmer.setCorr(obs_over_theo= tbl['HA6562_FLUX']/tbl['HB4861_FLUX'] / 2.86, wave1=6562.81, wave2=4861.33)
# set E(B-V) to zero if S/N is less than 3 in Halpha or Hbeta
rc_balmer.E_BV[(rc_balmer.E_BV<0) | (tbl['HB4861_FLUX']<3*tbl['HB4861_FLUX_ERR']) |  (tbl['HA6562_FLUX']<3*tbl['HA6562_FLUX_ERR'])] = 0
tbl['EBV_balmer'] = rc_balmer.E_BV
#tbl['A5007'] =  -2.5*np.log10(rc.getCorr(5007))

tbl['HA6562_FLUX_CORR'] = tbl['HA6562_FLUX'] * rc_balmer.getCorr(6562) 
tbl['HB4861_FLUX_CORR'] = tbl['HB4861_FLUX'] * rc_balmer.getCorr(4861)
tbl['HA6562_FLUX_CORR_ERR'] = tbl['HA6562_FLUX_ERR'] * rc_balmer.getCorr(6562) 
tbl['HB4861_FLUX_CORR_ERR'] = tbl['HB4861_FLUX_ERR'] * rc_balmer.getCorr(4861)

print('write to file')
primary_hdu = fits.PrimaryHDU()
table_hdu   = fits.BinTableHDU(tbl)
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto(f'{gal_name}_{HSTband}_ws{scalepc}pc_associations_Halpha.fits',overwrite=True)
