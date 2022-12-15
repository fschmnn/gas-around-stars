'''Find nearby associations to HII regions and measure their distance

For each mask in the nebulae catalogue, this script measures the 
distance to all associations in a size*size cutout, centered around
the HII region. The distance is measured from boundary to boundary.
The result is saved in a nested dictionary with the galaxy names as 
the first keys, then the region_ID and finally the assoc_ID. The 
values are the distance between nebula and association.
'''

from pathlib import Path
import numpy as np 
from tqdm import tqdm
import yaml

from astropy.coordinates import SkyCoord 
from astropy.table import Table 
import astropy.units as u 
from astropy.wcs import WCS 
from astropy.nddata import NDData, Cutout2D
from astropy.io import fits 

from reproject import reproject_interp
#from skimage.measure import find_contours
from scipy.spatial import distance

from cluster.io import read_associations
from astrotools.utils import resolution_from_wcs

version = 'v1p2'
HSTband = 'nuv'
scalepc = 32

basedir = Path('..')  # where we save stuff (and )
data_ext = Path('a:')/'Archive' # raw data

# size of the cutout
size = 20*u.arcsec

nebulae = Table.read(basedir / 'data' / 'interim' / 'Nebulae_Catalogue_v3.fits') 
nebulae['SkyCoord'] = SkyCoord(nebulae['cen_ra'],nebulae['cen_dec'],frame='icrs')

distances = {}

sample = ['IC5332','NGC0628','NGC1087','NGC1300','NGC1365',
          'NGC1385','NGC1433','NGC1512','NGC1566','NGC1672',
          'NGC2835','NGC3351','NGC3627','NGC4254','NGC4303',
          'NGC4321','NGC4535','NGC5068','NGC7496']

for gal_name in sample:
    filename = data_ext/'Products'/'Nebulae_catalogs'/'Nebulae_catalogue_v2'/'spatial_masks'/f'{gal_name}_nebulae_mask.fits'
    with fits.open(filename) as hdul:
        nebulae_mask = NDData(hdul[0].data.astype(float),meta=hdul[0].header,wcs=WCS(hdul[0].header))

    associations, associations_mask = read_associations(folder=data_ext/'Products'/'stellar_associations',target=gal_name.lower(),
                                                        HSTband=HSTband,scalepc=scalepc,version=version)
    
    pixel_to_arcsec = resolution_from_wcs(associations_mask.wcs)[0]

    assoc_IDs = np.unique(associations['assoc_ID'])
    region_IDs = np.unique(nebulae[nebulae['gal_name']==gal_name]['region_ID'])

    dic_neb = {}
    for row in tqdm(nebulae[nebulae['gal_name']==gal_name],desc=gal_name):
        
        region_ID = row['region_ID']
        position  = row['SkyCoord']

        cutout_assoc   = Cutout2D(associations_mask.data,position,size=size,wcs=associations_mask.wcs)
        cutout_nebulae = reproject_interp(nebulae_mask,output_projection=cutout_assoc.wcs,shape_out=cutout_assoc.shape,order='nearest-neighbor',return_footprint=False)    

        p1 = np.transpose(np.where(cutout_nebulae==region_ID))
        
        dic_asc = {}
        for assoc_ID in np.unique(cutout_assoc.data[~np.isnan(cutout_assoc.data)]):
            p2 = np.transpose(np.where(cutout_assoc.data==assoc_ID))
            dic_asc[int(assoc_ID)] = float(pixel_to_arcsec*distance.cdist(p1, p2).min())
        
        dic_neb[int(region_ID)] = dic_asc
        
    distances[gal_name] = dic_neb

    # we save the dictionary after each loop in case it crashes
    with open('neighboring_associations.yml','w+') as f:
        yaml.dump(distances,f)