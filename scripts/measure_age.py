from pathlib import Path

import numpy as np
from astropy.nddata import NDData
from astropy.io import fits 
from astropy.table import Table
from astropy.wcs import WCS

from datetime import date
from tqdm import tqdm
from photutils import CircularAperture,aperture_photometry

basedir = Path('..')
data_ext = Path('a:')/'Archive'

# the original catalogue from Francesco
with fits.open(basedir / 'data' / 'interim' / 'Nebulae_Catalogue_v2p1.fits') as hdul:
    nebulae = Table(hdul[1].data)
nebulae['age_mw'] = np.nan
nebulae['age_lw'] = np.nan
    
for gal_name in tqdm(np.unique(nebulae['gal_name'])):
    
    # read in the stellar pops ages
    filename = next((data_ext/'MUSE'/'DR2.1'/'copt').glob(f'{gal_name}*.fits'))
    copt_res = float(filename.stem.split('-')[1].split('asec')[0])
    with fits.open(filename) as hdul:
        age_mw = NDData(data=hdul['AGE_MW'].data,
                        meta=hdul['AGE_MW'].header,
                        wcs=WCS(hdul['AGE_MW'].header))
        age_lw = NDData(data=hdul['AGE_LW'].data,
                        meta=hdul['AGE_LW'].header,
                        wcs=WCS(hdul['AGE_LW'].header))

    tmp = nebulae[nebulae['gal_name']==gal_name]
    positions = np.transpose((tmp['cen_x'], tmp['cen_y']))
    apertures = CircularAperture(positions,2)
    ages_mw = aperture_photometry(age_mw,apertures)['aperture_sum'] / apertures.area     
    ages_lw = aperture_photometry(age_lw,apertures)['aperture_sum'] / apertures.area     

    nebulae['age_mw'][nebulae['gal_name']==gal_name] = ages_mw
    nebulae['age_lw'][nebulae['gal_name']==gal_name] = ages_lw

# write to file
columns = ['gal_name','region_ID','age_mw','age_lw']
    
doc = f'''this catalogue contains the ages from the stellar populations
fit measured at the position of the nebulae.
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
hdul.writeto('Nebulae_Catalogue_v2p1_age.fits',overwrite=True)