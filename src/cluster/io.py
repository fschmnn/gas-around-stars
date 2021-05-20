import numpy as np
from pathlib import Path 
from astropy.io import fits
from astropy.nddata import NDData
from astropy.wcs import WCS 
from astropy.table import Table


def read_associations(folder,target,scalepc,HSTband='nuv',version='v1p1'):
    '''read the catalogue and spatial mask for the associations
    
    Parameters
    ----------

    folder : pathlib.Path object
            the parent folder (HST/)
    
    target : string
            the name of the target
    
    scalepc : float
    
    HSTband : string (nuv/v)

    version : string
    '''
    
    folder = folder/f'associations {version}'/'multi-scale stellar associations'
    
    # define basefolder and check if file exists 
    if not (folder/target).is_dir():
        print(f'target not available. Use\n{",".join([x.stem for x in folder.iterdir()])}')
        raise FileNotFoundError
    target_folder = folder/target/f'{target}_{HSTband}_tracerstars'
    if not (target_folder/f'{scalepc}pc').is_dir():
        print(f'scalepc={scalepc} not available. Use:')
        print([int(x.stem[:-2]) for x in target_folder.iterdir() if x.stem.endswith('pc')])
        raise FileNotFoundError
    folder = target_folder/f'{scalepc}pc'

    # first the association catalogue
    catalogue_file = folder / f'{target}_phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits'
    with fits.open(catalogue_file) as hdul:
        associations = Table(hdul[1].data)

    # next the spatial masks for the associations
    mask_file = folder / f'{target}_phangshst_associations_{HSTband}_ws{scalepc}pc_idmask_{version}.fits'
    with fits.open(mask_file) as hdul:
        data = hdul[0].data.astype(float)
        data[data==0] = np.nan
        associations_mask = NDData(data,
                                   mask=data==0,
                                   meta=hdul[0].header,
                                   wcs=WCS(hdul[0].header))
    
    return associations, associations_mask
