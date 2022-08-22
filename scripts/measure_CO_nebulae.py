'''
 look for nearby GMC (or the CO) for nebulae
'''

from pathlib import Path 
import numpy as np
from datetime import date
from tqdm import tqdm

from astropy.io import fits, ascii
from astropy.table import Table 
from astropy.nddata import NDData
from astropy.wcs import WCS 
from astropy.coordinates import SkyCoord, match_coordinates_sky, Distance
import astropy.units as u 

from photutils import SkyCircularAperture, aperture_photometry


def get_value(matrix, index, default_value=np.nan):
    '''
    The `to_pixel` method returns the x,y coordinates. However in the 
    image they correspond to img[y,x]
    '''
    result = np.zeros(len(index))+default_value
    mask = (index[:,1] < matrix.shape[0]) & (index[:,0] < matrix.shape[1])
    mask &= (index[:,1] >= 0) & (index[:,0] >=0)

    valid = index[mask]
    result[mask] = matrix[valid[:,1], valid[:,0]]
    return result

basedir = Path('..')
data_ext = Path('a:')/'Archive'

# --------- define the location of the files --------------------------
nebulae_file = data_ext/'Products'/'Nebulae_catalogs'/'Nebulae_catalogue_v2'/ 'Nebulae_catalogue_v2.fits'

sample_table = ascii.read(basedir/'..'/'pnlf'/'data'/'interim'/'sample.txt')
sample_table.add_index('name')
sample_table['distance'] = Distance(distmod=sample_table['(m-M)'])

# the original catalogue from Francesco
with fits.open(nebulae_file) as hdul:
    nebulae = Table(hdul[1].data)
nebulae['SkyCoord'] = SkyCoord(nebulae['cen_ra']*u.deg,nebulae['cen_dec']*u.deg,frame='icrs')

nebulae = nebulae[['gal_name','region_ID','SkyCoord']]
nebulae['CO'] = np.nan 
nebulae['GMC_sep'] = np.nan
nebulae['GMC_random_sep'] = np.nan
nebulae['GMC_mass'] = np.nan


for gal_name in tqdm(np.unique(nebulae['gal_name'])):

    # first the CO
    co_filename = data_ext/'ALMA'/'v4p0'/f'{gal_name.lower()}_12m+7m+tp_co21_broad_tpeak.fits'
    if not co_filename.is_file():
        print(f'no ALMA CO for {gal_name}')
    else:
        with fits.open(co_filename) as hdul:
            CO = NDData(data=hdul[0].data,
                        meta=hdul[0].header,
                        wcs=WCS(hdul[0].header))

        apertures = SkyCircularAperture(nebulae[nebulae['gal_name']==gal_name]['SkyCoord'],r=4*u.arcsec)
        nebulae['CO'][nebulae['gal_name']==gal_name] = aperture_photometry(CO,apertures)['aperture_sum']

    # and then the GMC catalogue
    gmc_directory  = data_ext/'Products'/'GMC'/'matched'
    gmc_resolution = '150pc'
    gmc_filename = [x for x in (gmc_directory/gmc_resolution).iterdir() if gal_name.lower() in x.stem]
    if len(gmc_filename)==0:
        print(f'no GMC catalogue for {gal_name}')
        continue

    gmc_filename = gmc_filename[0]
    with fits.open(gmc_filename) as hdul:
        GMC = Table(hdul[1].data)
    GMC['SkyCoord'] = SkyCoord(GMC['XCTR_DEG']*u.deg,GMC['YCTR_DEG']*u.deg)

    idx,sep,_=match_coordinates_sky(nebulae[nebulae['gal_name']==gal_name]['SkyCoord'],GMC['SkyCoord'])
    nebulae['GMC_sep'][nebulae['gal_name']==gal_name] = sep.to(u.arcsec)
    nebulae['GMC_mass'][nebulae['gal_name']==gal_name] = GMC[idx]['MLUM_MSUN']

    # create random points and make sure they are in the ALMA FOV
    N_HII = np.sum(nebulae['gal_name']==gal_name)
    points = np.random.uniform(low=(0,0),high=CO.data.shape,size=(2*N_HII,2))

    value = get_value(CO.data,points.astype(int))
    x,y = points[~np.isnan(value)][:N_HII].T
    coords = SkyCoord.from_pixel(x,y,wcs=CO.wcs)
    
    idx,sep,_=match_coordinates_sky(coords,GMC['SkyCoord'])
    nebulae['GMC_random_sep'][nebulae['gal_name']==gal_name] = np.mean(sep).to(u.arcsec)

# nebulae outside the ALMA FOV are set to nan
nebulae['GMC_sep'][np.isnan(nebulae['CO'])] = np.nan 


# write to file
columns = ['gal_name','region_ID','CO','GMC_sep','GMC_random_sep','GMC_mass']
    
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
table_hdu   = fits.BinTableHDU(nebulae[columns])
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto(f'Nebulae_Catalogue_v2p1_CO.fits',overwrite=True)