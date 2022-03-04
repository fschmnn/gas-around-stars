'''
to judge wheather the ages of the associations are trustworthy, 
we look for nearby GMC (or the CO)
'''

from pathlib import Path 
import numpy as np
from datetime import date
from tqdm import tqdm

from astropy.io import fits 
from astropy.table import Table 
from astropy.nddata import NDData
from astropy.wcs import WCS 
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u 

from photutils import SkyCircularAperture, aperture_photometry



# choose which version of the association catalogue to use
version = 'v1p2'
HSTband = 'nuv'
scalepc = 32

basedir = Path('..')
data_ext = Path('a:')/'Archive'

# the association catalogue
with fits.open(basedir/'data'/'interim'/f'phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits') as hdul:
    associations = Table(hdul[1].data)
associations['SkyCoord'] = SkyCoord(associations['reg_ra']*u.degree,associations['reg_dec']*u.degree)
associations = associations[['gal_name','assoc_ID','SkyCoord']]
associations['CO'] = np.nan 
associations['GMC_sep'] = np.nan

for gal_name in tqdm(np.unique(associations['gal_name'])):

    # first the CO
    co_filename = data_ext/'ALMA'/'v4p0'/f'{gal_name.lower()}_12m+7m+tp_co21_broad_tpeak.fits'
    if not co_filename.is_file():
        print(f'no ALMA CO for {gal_name}')
    else:
        with fits.open(co_filename) as hdul:
            CO = NDData(data=hdul[0].data,
                        meta=hdul[0].header,
                        wcs=WCS(hdul[0].header))

        apertures = SkyCircularAperture(associations[associations['gal_name']==gal_name]['SkyCoord'],r=4*u.arcsec)
        associations['CO'][associations['gal_name']==gal_name] = aperture_photometry(CO,apertures)['aperture_sum']

    # and then the GMC catalogue
    gmc_directory  = data_ext/'Products'/'GMC'/'matched'
    gmc_resolution = '150pc'
    gmc_filename = [x for x in (gmc_directory/gmc_resolution).iterdir() if gal_name.lower() in x.stem]
    if len(gmc_filename)==0:
        print(f'no GMC catalogue for {gal_name}')
    else:
        gmc_filename = gmc_filename[0]
        with fits.open(gmc_filename) as hdul:
            GMC = Table(hdul[1].data)
        GMC['SkyCoord'] = SkyCoord(GMC['XCTR_DEG']*u.deg,GMC['YCTR_DEG']*u.deg)

        idx,sep,_=match_coordinates_sky(associations[associations['gal_name']==gal_name]['SkyCoord'],GMC['SkyCoord'])
        associations['GMC_sep'][associations['gal_name']==gal_name] = sep.to(u.arcsec)


    # write to file
columns = ['gal_name','assoc_ID','CO','GMC_sep']
    
doc = f'''this catalogue contains the CO measured at the position
of the association and the distance to the nearest GMC.
last update: {date.today().strftime("%b %d, %Y")}
'''

primary_hdu = fits.PrimaryHDU()
for i,comment in enumerate(doc.split('\n')):
    if i==0:
        primary_hdu.header['COMMENT'] = comment
    else:
        primary_hdu.header[''] = comment
table_hdu   = fits.BinTableHDU(associations[columns])
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto(f'association_CO_{HSTband}_ws{scalepc}pc_{version}.fits',overwrite=True)