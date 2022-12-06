'''Construct a catalogue of HII region complexes


Output files 
------------

nebulae_neighbors.yml : dictionary
    the neighbouring nebulae for all nebulae

complexes_nebulae.yml : dictionary
    for each complex, this dictionary lists all nebulae that are a part of it

complexes_associations_version  :dictionary
    for each complex, this dictionary lists all contained associations

complexes.fits : fits table
    the main output file. 

complexes_nebulae.fits : fits table
    to conveniently add the `complex_ID` to the nebulae by joining
    the original catalogue with this table.

complexes_associations.fits : fits table
    to conveniently add the `complex_ID` to the associations by joining
    the original catalogue with this table.
'''

# ---------------------------------------------------------------------
# basic os stuff
# ---------------------------------------------------------------------
from pathlib import Path       # use instead of os.path and glob
from tqdm import tqdm          # progress bar
from datetime import date 
import yaml

# ---------------------------------------------------------------------
# datastructures and scientific computing
# ---------------------------------------------------------------------
import numpy as np                       # arrays
from skimage.segmentation import find_boundaries

# ---------------------------------------------------------------------
# astronomy related stuff
# ---------------------------------------------------------------------

from astropy.table import Table          # useful datastructure
from astropy.table import vstack, join   # combine multiple tables
from astropy.nddata import NDData, StdDevUncertainty, Cutout2D  
from astropy.io import fits        # open text and fits files
from astropy.wcs import WCS              # handle coordinates
from astropy.coordinates import SkyCoord # convert pixel to sky coordinates

# other stuff
import networkx as nx
from skimage.measure import moments              

version = 'v1p2'
HSTband = 'nuv'
scalepc = 32

def nanunique(data):
    return np.unique(data[~np.isnan(data)])

def error_prop(array):
    return np.sqrt(np.sum(array**2))

basedir  = Path('..')
data_ext = Path('a:')/'Archive'

# read in the nebulae catalogue
nebulae = Table.read(basedir / 'data' / 'interim' / 'Nebulae_Catalogue_v3.fits')
nebulae['complex_ID'] = np.nan
nebulae['isHII'] = (nebulae['BPT_NII']==0) & (nebulae['BPT_SII']==0) & (nebulae['BPT_OI']==0) & (nebulae['flag_star']==0) & (nebulae['flag_edge']==0)

# and the associations
associations = Table.read(basedir/'data'/'interim'/f'phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits') 
associations['complex_ID'] = np.nan

nebulae_neighbors = {}
complexes_neb   = {}
complexes_assoc = {}
complexes_lst = []
for gal_name in tqdm(np.unique(nebulae['gal_name'])):

    # read in the data
    filename = data_ext / 'MUSE' / 'DR2.1' / 'copt' / 'MUSEDAP'
    filename = [x for x in filename.iterdir() if x.stem.startswith(gal_name)][0]
    with fits.open(filename) as hdul:
        Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                        uncertainty=StdDevUncertainty(hdul['HA6562_FLUX_ERR'].data),
                        mask=np.isnan(hdul['HA6562_FLUX'].data),
                        meta=hdul['HA6562_FLUX'].header,
                        wcs=WCS(hdul['HA6562_FLUX'].header))

    filename = data_ext / 'Products' / 'Nebulae_catalogs'/'Nebulae_catalogue_v2' /'spatial_masks'/f'{gal_name}_nebulae_mask.fits'
    with fits.open(filename) as hdul:
        nebulae_mask = NDData(hdul[0].data.astype(float),mask=Halpha.mask,meta=hdul[0].header,wcs=WCS(hdul[0].header))
        nebulae_mask.data[nebulae_mask.data==-1] = np.nan 

    # we also load the overlap dictionaries
    with open(basedir/'data'/'map_nebulae_association'/version/HSTband/f'{scalepc}pc'/f'{gal_name}_{HSTband}_{scalepc}pc_nebulae.yml') as f:
        nebulae_dict = yaml.load(f,Loader=yaml.SafeLoader)
    with open(basedir/'data'/'map_nebulae_association'/version/HSTband/f'{scalepc}pc'/f'{gal_name}_{HSTband}_{scalepc}pc_associations.yml') as f:
        associations_dict = yaml.load(f,Loader=yaml.SafeLoader)

    neighbors = {}
    for row in nebulae[nebulae['gal_name']==gal_name]:
        radius = np.sqrt(row['region_area']/np.pi)
        cutout = Cutout2D(nebulae_mask.data,(row['cen_x'],row['cen_y']),size=(3*radius,3*radius),mode='partial',fill_value=np.nan)
        # we need to convert the entries to int in order to save them to a yaml file
        lst = list(nanunique(cutout.data[find_boundaries(cutout.data==row['region_ID'],mode='outer')]).astype(int))
        if len(lst)>0:
            neighbors[int(row['region_ID'])] = [int(x) for x in lst]
    
    nebulae_neighbors[gal_name] = neighbors

    # we use networkx to group HII regions toegether
    G = nx.Graph()
    for k,v in neighbors.items():
        for j in v:
            G.add_edge(k,j)

    # this list of lists contains all complexes and the HII regions that constitute each complex
    neb_list = sorted(nx.connected_components(G), key = min, reverse=False)
    HII_complex = Table({'gal_name' : [gal_name]*len(neb_list),
                        'complex_ID': list(map(min,neb_list)),
                        'region_IDs': neb_list})
    # again we need to ensure that all entries are int in order save as yaml
    complexes_neb[gal_name] = {int(row['complex_ID']):[int(x) for x in row['region_IDs']] for row in HII_complex}

    # a list of lists that contains the associations in each complex 
    assoc_list = []
    valid_list = []
    assoc_dic = {}
    for row in HII_complex: 
        assoc_IDs = set()
        valid = True
        for region_ID in row['region_IDs']:
            assoc_IDs.update(nebulae_dict[region_ID])
        
        # we check if the association is contained in a single complex
        for assoc_ID in assoc_IDs:
            if np.all(np.isin(list(associations_dict[assoc_ID]),list(row['region_IDs']))):
                associations['complex_ID'][(associations['gal_name']==gal_name) & (associations['assoc_ID']==assoc_ID)] = row['complex_ID']
            else:
                valid = False

        assoc_dic[int(row['complex_ID'])] = [int(x) for x in assoc_IDs]
        assoc_list.append(assoc_IDs) 
        valid_list.append(valid)

    HII_complex['assoc_IDs'] = assoc_list

    # check if all objects are HII regions
    HII_complex['isHII'] = False
    for row in HII_complex:
        if np.all(nebulae[(nebulae['gal_name']==gal_name) & np.isin(nebulae['region_ID'],list(row['region_IDs']))]['isHII']):
            row['isHII'] = True


    HII_complex['multi_to_multi'] = valid_list
    complexes_assoc[gal_name] = assoc_dic

    # compute some properties of the complexes 
    HII_complex['x']   = np.nan
    HII_complex['y']   = np.nan
    HII_complex['ra']  = np.nan
    HII_complex['dec'] = np.nan

    for row in HII_complex:
        # find the center of the complex
        M = moments(np.isin(nebulae_mask.data,list(row['region_IDs'])))
        y,x = M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]
        coord = SkyCoord.from_pixel(x,y,nebulae_mask.wcs)
        row[['x','y','ra','dec']] = x,y,coord.ra.value,coord.dec.value

    # add information to the nebulae catalogue
    for row in HII_complex:
        for region_ID in row['region_IDs']:
            nebulae['complex_ID'][(nebulae['gal_name']==gal_name) & (nebulae['region_ID']==region_ID)] = row['complex_ID']


    complexes_lst.append(HII_complex)

# we save the dictionary with the exact neighboring information
with open(basedir/'data'/'interim'/f'nebulae_neighbors.yml','w+') as f:
    yaml.dump(nebulae_neighbors,f)
with open(basedir/'data'/'interim'/f'complexes_nebulae.yml','w+') as f:
    yaml.dump(complexes_neb,f)
with open(basedir/'data'/'interim'/f'complexes_associations_{HSTband}_ws{scalepc}pc_{version}.yml','w+') as f:
    yaml.dump(complexes_assoc,f)

complexes = vstack(complexes_lst)
# we can not save the column as an object
complexes['region_IDs'] = [','.join(map(str,x)) for x in complexes['region_IDs']] 
complexes['assoc_IDs'] = [','.join(map(str,x)) for x in complexes['assoc_IDs']] 


# write the catalogues to a file
complexes.write(basedir / 'data' / 'interim' / 'complexes.fits',overwrite=True)
nebulae['gal_name','region_ID','complex_ID'].write(basedir / 'data' / 'interim' / 'complexes_nebulae.fits',overwrite=True)
associations['gal_name','assoc_ID','complex_ID'].write(basedir/'data'/'interim'/f'complexes_associations_{HSTband}_ws{scalepc}pc_{version}.fits',overwrite=True) 

print('job done')