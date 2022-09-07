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
    
# define the intervals for EW(Halpha) and EW(Hbeta)
Halpha_interval=[6557.6,6571.35]
continuum_Halpha_interval=[(6483,6513),(6623,6653)]
    
Hbeta_interval = [4847.9,4876.6]
continuum_Hbeta_interval = [(4827.9,4847.9),(4876.6,4891.6)]

# read the nebulae catalogue
with fits.open(Path('/data')/'Products'/'Nebulae_catalogs'/'Nebulae_catalogue_v2'/'Nebulae_catalogue_v2.fits') as hdul:
    catalogue = Table(hdul[1].data)
tmp = catalogue['gal_name','region_ID','HA6562_VEL','HA6562_FLUX','HA6562_FLUX_ERR'].copy()

# define the new columns
tmp['Halpha'] = np.nan
tmp['Halpha_error'] = np.nan
tmp['Halpha_bkg'] = np.nan
tmp['Halpha_bkg_error'] = np.nan
tmp['continuum_Halpha'] = np.nan
tmp['continuum_Halpha_error'] = np.nan
tmp['continuum_Halpha_bkg'] = np.nan
tmp['continuum_Halpha_bkg_error'] = np.nan

tmp['Hbeta'] = np.nan
tmp['Hbeta_error'] = np.nan
tmp['Hbeta_bkg'] = np.nan
tmp['Hbeta_bkg_error'] = np.nan
tmp['continuum_Hbeta'] = np.nan
tmp['continuum_Hbeta_error'] = np.nan
tmp['continuum_Hbeta_bkg'] = np.nan
tmp['continuum_Hbeta_bkg_error'] = np.nan

    
# the systematic velocities of the galaxies from Groves et al. (in preparation)
v_sys = {'IC5332': 699.0,'NGC0628': 651.0,'NGC1087': 1502.0,'NGC1300': 1545.0,'NGC1365': 1613.0,
         'NGC1385': 1477.0,'NGC1433': 1057.0,'NGC1512': 871.0,'NGC1566': 1483.0,'NGC1672': 1318.0,
         'NGC2835': 867.0,'NGC3351': 775.0,'NGC3627': 715.0,'NGC4254': 2388.0,'NGC4303': 1560.0,
         'NGC4321': 1572.0,'NGC4535': 1954.0,'NGC5068': 667.0,'NGC7496': 1639.0}

# Milky Way E(B-V) from  Schlafly & Finkbeiner (2011)
EBV_MW = {'IC5332': 0.015,'NGC0628': 0.062,'NGC1087': 0.03,'NGC1300': 0.026,
          'NGC1365': 0.018,'NGC1385': 0.018,'NGC1433': 0.008,'NGC1512': 0.009,
          'NGC1566': 0.008,'NGC1672': 0.021,'NGC2835': 0.089,'NGC3351': 0.024,
          'NGC3627': 0.037,'NGC4254': 0.035,'NGC4303': 0.02,'NGC4321': 0.023,
          'NGC4535': 0.017,'NGC5068': 0.091,'NGC7496': 0.008}
    
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
    return hii_region, continuum*area_mask


def measure_ew(spectrum,error,wlam,z,line_interval=[6557.6,6571.35],
               continuum_interval=[(6483,6513),(6623,6653)],plot=False):
    '''measure the equivalent width of Halpha (or any other line)
    
    
    Parameters 
    ----------
    spectrum : 
        the spectrum
    wlam : 
        spectral axis
    z :
        redshift
    line_interval :
        range to sum up for line flux (default is for Halpha)
    continuum_interval :
        range to sum up for continuum (default is for Halpha)
        
    
    Returns
    -------
     line_flux, : 
        the summed emission line flux
    line_flux_error :
        error of the summed emission line flux
    continuum : 
        the summed continuum
    continuum_error : 
        error of the summed continuum
    '''
    
    mask = (wlam>(1+z)*line_interval[0]) & (wlam<(1+z)*line_interval[1])
    continuum_mask = np.zeros_like(wlam,dtype=bool)
    for interval in continuum_interval:
        continuum_mask |= ((wlam>(1+z)*interval[0]) & (wlam<(1+z)*interval[1]))
    channel_width = 1.2503324

    continuum = sigma_clipped_stats(spectrum[continuum_mask])[0] 
    #continuum_error = np.std(spectrum[continuum_mask])
    continuum_error = np.sqrt(np.sum(error[continuum_mask]**2)) / np.sum(continuum_mask)
    
    line_flux = np.sum(spectrum[mask]-continuum) * channel_width 
    line_flux_error = np.sqrt(np.sum(error[mask]**2)) * channel_width 
    
    if plot:
        fig,ax=plt.subplots(figsize=(12,4))
        ax.axhline(0,color='black',lw=0.5)
        ax.plot(wlam,spectrum-continuum,color='tab:green')
        ax.axvline((1+z)*6562,color='gray')
        for interval in continuum_interval:
            ax.axvspan((1+z)*interval[0],(1+z)*interval[1], color='gray', alpha=0.5, lw=0)
        ax.axvspan((1+z)*line_interval[0],(1+z)*line_interval[1], color='gray', alpha=0.5, lw=0)
        ax.set(xlabel=r'$\lambda$ / \AA',ylim=[None,None],xlim=[6450,6700],yscale='linear')
        plt.show()
    
    return line_flux, line_flux_error, continuum, continuum_error


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
            error_cube = hdul[2].data
        # this is the same for all nebulae
        wlam = np.linspace(cube_header['CRVAL3'],cube_header['CRVAL3']+cube_header['NAXIS3']*cube_header['CD3_3'],cube_header['NAXIS3'])
        # we also need to update the extinction correction
        RedCorr = pn.RedCorr(R_V=3.1,E_BV=EBV_MW[gal_name],law='CCM89 oD94')

    # determine the redshift of the current nebulae
    z = ((v_sys[gal_name]+row['HA6562_VEL'])*u.km/u.s/c.c).decompose().value

    spec,bkg = extract_spectrum_with_background(data_cube,nebulae_mask,row['region_ID'],factor=factor)
    error_spec, error_bkg_spec = extract_spectrum_with_background(error_cube,nebulae_mask,row['region_ID'],factor=factor)

    # we need to correct for MW extinction
    spec *= RedCorr.getCorr(wlam)
    bkg  *= RedCorr.getCorr(wlam)
    error_spec *= RedCorr.getCorr(wlam)
    error_bkg_spec *= RedCorr.getCorr(wlam)
    
    # measure the equivalent width of Halpha
    Halpha, Halpha_error, continuum, continuum_error = measure_ew(spec,error_spec,wlam,z)
    Halpha_bkg, Halpha_bkg_error, continuum_bkg, continuum_bkg_error = measure_ew(bkg,error_bkg_spec,wlam,z)

    row['continuum_Halpha'] = continuum
    row['continuum_Halpha_error'] = continuum_error
    row['continuum_Halpha_bkg'] = continuum_bkg
    row['continuum_Halpha_bkg_error'] = continuum_bkg_error
    row['Halpha'] = Halpha
    row['Halpha_error'] = Halpha_error
    row['Halpha_bkg'] = Halpha_bkg
    row['Halpha_bkg_error'] = Halpha_bkg_error
    
    # repeat the last step for Hbeta
    Hbeta, Hbeta_error, continuum, continuum_error = measure_ew(spec,error_spec,wlam,z,
                                                                line_interval=Hbeta_interval,
                                                                continuum_interval=continuum_Hbeta_interval)
    Hbeta_bkg, Hbeta_bkg_error, continuum_bkg, continuum_bkg_error = measure_ew(bkg,error_bkg_spec,wlam,z,
                                                                                line_interval=Hbeta_interval,
                                                                                continuum_interval=continuum_Hbeta_interval)
    
    row['continuum_Hbeta'] = continuum
    row['continuum_Hbeta_error'] = continuum_error
    row['continuum_Hbeta_bkg'] = continuum_bkg
    row['continuum_Hbeta_bkg_error'] = continuum_bkg_error
    row['Hbeta'] = Hbeta
    row['Hbeta_error'] = Hbeta_error
    row['Hbeta_bkg'] = Hbeta_bkg    
    row['Hbeta_bkg_error'] = Hbeta_bkg_error
    
doc = f'''measure equivalent widths for the nebulae catalogue  
EW(Ha) and EW(Hb) are measured with the following steps:
1) The integrated spectra of each HII region is extracted from the 
   native cube. For the continuum, the median along the spatial axis is
   used. To estimate the background, an annulus that is 3 times the 
   area of the HII region is created and its spectra is also extracted.
2) All spectra are corrected for Milky Way extinction.
3) The continuum is estimated from the sigma clipped mean in the range
   6483.0 to 6513.0 AA and 6623.0 to 6653.0 AA for Halpha and 
   4827.9 to 4847.9 AA and 4876.6 to 4891.6 AA for Hbeta. The flux of 
   the emission lines is measured by summing the continuum subtracted
   spectra in the range of 6557.6 to 6571.35 AA in the case of Halpha
   and 4847.9 to 4876.6 AA for Hbeta. For both the continuum and the 
   emission line flux, the background emission is estimated from the 
   annulus in the same way. The uncertainty of each property is 
   estimated from the variance cube
Based on `Nebulae_catalogue_v2.fits`. 
This catalogue was created with the following script:
https://github.com/fschmnn/cluster/blob/master/scripts/measure_eq_width.py
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
hdul.writeto('Nebulae_Catalogue_v2p1_EW.fits',overwrite=True)
    