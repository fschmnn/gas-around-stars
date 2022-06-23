'''Match nebulae and association catalogues

This script reads in all the neccesarry files and matches the two catalogues

Parameters 
----------
name / -n
    The name of the galaxy (use all to run all galaxies)
version / -v
    The version of the association catalogue (only works for v1p2 at the moment)
HSTband / -b
    The band that is used for the tracers (v or nuv)
scalepc / -s
    The scale of the associations (8,16,32 or 64)
'''

# ---------------------------------------------------------------------
# basic os stuff
# ---------------------------------------------------------------------
from pathlib import Path       # use instead of os.path and glob
import logging                 # use logging instead of print
import sys
from tqdm import tqdm          # progress bar

# ---------------------------------------------------------------------
# datastructures and scientific computing
# ---------------------------------------------------------------------
import numpy as np                       # arrays

# ---------------------------------------------------------------------
# astronomy related stuff
# ---------------------------------------------------------------------

from astropy.table import Table,QTable   # useful datastructure
from astropy.nddata import NDData, StdDevUncertainty, Cutout2D  
from astropy.io import ascii,fits        # open text and fits files
from astropy.wcs import WCS              # handle coordinates
from astropy.coordinates import SkyCoord # convert pixel to sky coordinates
import astropy.units as u

from astrotools.regions import find_sky_region
from cluster.io import ReadHST

scalepc = 16  
HSTband = 'v'
version = 'v1p2'

logging.basicConfig(stream=sys.stdout,datefmt='%H:%M:%S',level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('astropy').setLevel(logging.WARNING)

basedir  = Path('..')
data_ext = Path('a:')/'Archive'

# --------- define the location of the files --------------------------
nebulae_file = data_ext/'Products'/'Nebulae_catalogs'/'Nebulae_catalogue_v2'/ 'Nebulae_catalogue_v2.fits'
association_file = basedir / 'data' / 'interim' / f'phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits'
# ---------------------------------------------------------------------

# the original catalogue from Francesco
with fits.open(nebulae_file) as hdul:
    nebulae = Table(hdul[1].data)
nebulae['SkyCoord'] = SkyCoord(nebulae['cen_ra']*u.deg,nebulae['cen_dec']*u.deg,frame='icrs')
nebulae['in_frame'] = False

with fits.open(association_file) as hdul:
    associations = Table(hdul[1].data)
associations['SkyCoord'] = SkyCoord(associations['reg_ra']*u.degree,associations['reg_dec']*u.degree)
associations['in_frame'] = False


for gal_name in tqdm(np.unique(nebulae['gal_name'])):
    
    filename = data_ext / 'MUSE' / 'DR2.1' / 'copt' / 'MUSEDAP'
    filename = [x for x in filename.iterdir() if x.stem.startswith(gal_name)][0]

    with fits.open(filename) as hdul:
        Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                        uncertainty=StdDevUncertainty(hdul['HA6562_FLUX_ERR'].data),
                        mask=np.isnan(hdul['HA6562_FLUX'].data),
                        meta=hdul['HA6562_FLUX'].header,
                        wcs=WCS(hdul['HA6562_FLUX'].header))

    # the association catalogue and mask
    target  = gal_name.lower()

    if gal_name not in associations['gal_name']:
        print(f'{gal_name} not in association catalogue')
        continue

    hst_images = ReadHST(gal_name,data_ext/'HST'/'filterImages',HSTbands=['f275w'])
    f275w = hst_images.f275w

    reg_muse_pix, reg_muse_sky = find_sky_region(Halpha.mask.astype(int),wcs=Halpha.wcs)
    reg_hst_pix, reg_hst_sky = find_sky_region(f275w.mask.astype(int),wcs=f275w.wcs)

    # check which nebulae/clusters are within the HST/MUSE FOV
    associations['in_frame'][associations['gal_name']==gal_name] = reg_muse_sky.contains(associations['SkyCoord'][associations['gal_name']==gal_name],Halpha.wcs)
    #clusters['in_frame'] = reg_muse_sky.contains(clusters['SkyCoord'],nebulae_mask.wcs)
    nebulae['in_frame'][nebulae['gal_name']==gal_name] = reg_hst_sky.contains(nebulae['SkyCoord'][nebulae['gal_name']==gal_name],f275w.wcs)

print(f'{np.sum(associations["in_frame"])} (of {len(associations)}) associations in MUSE FOV')
#print(f'{np.sum(clusters["in_frame"])} (of {len(clusters)}) clusters in MUSE FOV')
print(f'{np.sum(nebulae["in_frame"])} (of {len(nebulae)}) nebulae in HST FOV')


hdu = fits.BinTableHDU(nebulae[['gal_name','region_ID','in_frame']],name='in_frame')
hdu.writeto(basedir/'data'/'interim'/f'Nebulae_Catalogue_v2p1_in_frame.fits',overwrite=True)

hdu = fits.BinTableHDU(associations[['gal_name','assoc_ID','in_frame']],name='in_frame')
hdu.writeto(basedir/'data'/'interim'/f'phangshst_associations_{HSTband}_ws{scalepc}pc_{version}_in_frame.fits',overwrite=True)

