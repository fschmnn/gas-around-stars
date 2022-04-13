from pathlib import Path 

import re
import logging 
import numpy as np

from astropy.io import fits
from astropy.nddata import NDData
from astropy.wcs import WCS 
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u 
from astropy.units import spectral_density
from astropy.visualization import simple_norm

import matplotlib.pyplot as plt 


logger = logging.getLogger(__name__)
HSTbands_wave = {'NUV':2704*u.AA,'U':3355*u.AA,'B':4325*u.AA,'V':5308*u.AA,'I':8024*u.AA}
freq_to_wave = lambda band: u.mJy.to(u.erg/u.s/u.cm**2/u.Angstrom,equivalencies=spectral_density(HSTbands_wave[band]))


class ReadHST:
    '''A class to read HST images in multiple filters'''
        
    def __init__(self,name,directory,HSTbands=['f275w','f336w','f435w','f438w','f555w','f814w']):
        '''
        Parameters 
        ----------
        
        name : string
        
        directory : Path object
        
        HSTbands : list
        '''
        
        logger.info(f'reading {name}')
        self.name = name

        for HSTband in HSTbands:
            pattern = re.compile(f'(?=.*{name.lower()})(?=.*{HSTband})(?=.*exp-drc-sci)')
            filename = [file for file in directory.iterdir() if pattern.search(file.stem)]
            if len(filename)==0:
                logger.warning(f'no file for {HSTband}')
            else:
                with fits.open(filename[0]) as hdul:
                    data = NDData(hdul[0].data,
                                  mask=hdul[0].data==0,
                                  meta=hdul[0].header,
                                  wcs=WCS(hdul[0].header))
                setattr(self,HSTband,data)
        
    def __repr__(self):
        '''create an overview of the available attributes'''
   
        string = ''
        for k,v in self.__dict__.items():
            if type(v) in [str,int,float]:
                string += f'{k}: {v}\n'
            else:
                string += k + '\n'
        return string
    
    def plot(self,HSTband,figsize=(6,6),cmap=plt.cm.hot,**kwargs):
        
        if not hasattr(self,HSTband):
            raise AttributeError(f'object has no attribute {HSTband}')
                
        data = getattr(self,HSTband)
        
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(projection=data.wcs)
        norm = simple_norm(data.data,clip=False,percent=99)
        ax.imshow(data.data,norm=norm,cmap=cmap)
        ax.set(**kwargs)
        
        plt.show()

def read_associations(folder,target,scalepc,HSTband='nuv',version='v1p2',data='all'):
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

    data : string
        'all', 'catalogue', 'mask'
    '''
    
    folder = folder/f'associations_{version}'
    
    # define basefolder and check if file exists 
    if not (folder/f'{target}_{HSTband}').is_dir():
        print(f'target not available. Use\n{",".join([x.stem for x in folder.iterdir()])}')
        raise FileNotFoundError
    target_folder = folder/f'{target}_{HSTband}'
    if not (target_folder/f'{scalepc}pc').is_dir():
        msg = f'scalepc={scalepc}pc not available. Use: '
        msg += ','.join([x.stem for x in target_folder.iterdir() if x.stem.endswith('pc')])
        raise FileNotFoundError(msg)
    folder = target_folder/f'{scalepc}pc'

    if data=='all' or data=='catalogue':
        # first the association catalogue
        catalogue_file = folder / f'{target}_phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits'
        with fits.open(catalogue_file) as hdul:
            associations = Table(hdul[1].data)

        # modify table (rename the columns such that the clusters and associations are identical)
        associations['SkyCoord'] = SkyCoord(associations['reg_ra']*u.degree,associations['reg_dec']*u.degree)
        associations.rename_columns(['reg_id','reg_ra','reg_dec','reg_x','reg_y',
                                    'reg_dolflux_Age_MinChiSq','reg_dolflux_Mass_MinChiSq','reg_dolflux_Ebv_MinChiSq',
                                    'reg_dolflux_Age_MinChiSq_err','reg_dolflux_Mass_MinChiSq_err','reg_dolflux_Ebv_MinChiSq_err'],
                                    ['assoc_ID','RA','DEC','X','Y','age','mass','EBV','age_err','mass_err','EBV_err'])
        #for col in list(associations.columns):
        #    if col.endswith('mjy'):
        #        associations[f'{col.split("_")[0]}_FLUX'] = 1e20*associations[col]*u.mJy.to(u.erg/u.s/u.cm**2/u.Hz)
        #    if col.endswith('mjy_err'):
        #        associations[f'{col.split("_")[0]}_FLUX_ERR'] = 1e20*associations[col]*u.mJy.to(u.erg/u.s/u.cm**2/u.Hz)

        for col in list(associations.columns):
            if col.endswith('mjy'):
                band = col.split("_")[0]
                associations[f'{band}_FLUX'] = associations[col]*freq_to_wave(band)
                associations[f'{band}_FLUX_ERR'] = associations[col+'_err']*freq_to_wave(band)

        if data=='catalogue':
            return associations

    if data=='all' or data=='mask':
        # next the spatial masks for the associations
        mask_file = folder / f'{target}_phangshst_associations_{HSTband}_ws{scalepc}pc_idmask_{version}.fits'
        with fits.open(mask_file) as hdul:
            mask = hdul[0].data.astype(float)
            mask[mask==0] = np.nan
            associations_mask = NDData(mask,
                                    mask=mask==0,
                                    meta=hdul[0].header,
                                    wcs=WCS(hdul[0].header))
        if data=='mask':
            return associations_mask

    return associations, associations_mask
