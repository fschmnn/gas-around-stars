'''measure Halpha/Hbeta and E(B-V) from MUSE for the cluster catalogue

1. Read the Halpha and Hbeta linemap from the MUSEDAP
2. Read the compact cluster catalogue
3. Measure the background subtracted fluxes 
4. The fluxes are corrected for Milkyway extinction
5. The Internal reddening is estimated from the Balmer decrement
6. The fluxes are corrected for internal extinction (_FLUX_CORR)
7. The catalogue is saved to a file
'''

from datetime import date
doc = f'''Measure Halpha from MUSE for compact clusters

This folder contains Halpha fluxes that were measured for the 
compact cluster catalogues. The flux is measured in an aperture 
with a fixed size of 0.8" and the background is estimate from the
sigma clipped median within an annulus (inner radius 1.5" and outer
radius 2"). All fluxes are corrected for Milky Way extinction with 
the O'Donnell (1994) extinction curve and E(B-V) from  Schlafly & 
Finkbeiner (2011). The internal extinction is calculated from the
Balmer decrement (assuming a theoretical ratio Ha/Hb=2.86).

The full catalogues can be found here
https://drive.google.com/drive/folders/1L_wLR0_7bNifp8DSqW4kUrb-YTAAYGDq
date created: {date.today().strftime("%b %d, %Y")}
'''    


from pathlib import Path      # handle paths to files
from tqdm import tqdm         # progress bar for long loops
import numpy as np                # arrays and useful math functions

from astropy.io import fits     # read/write from/to files
from astropy.table import Table        # useful data structure
from astropy.coordinates import SkyCoord    # same
from astropy.wcs import WCS
import astropy.units as u              # convert between different units
from astropy.nddata import NDData, StdDevUncertainty # useful when working with data

import pyneb    # emission line stuff (used for extinction correction)

from photutils import SkyCircularAperture, SkyCircularAnnulus, ApertureStats
from astropy.stats import SigmaClip

# we use this function to check if a point is in the FOV
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

r     = .8*u.arcsec
r_in  = 1.5*u.arcsec
r_out = 2*u.arcsec

# the folder with the data (structure similar to Google Drive)
data_ext = Path('a:')/'Archive'
out_folder = Path('.')/'compact_clusters_Halpha'

# Milky Way E(B-V) from  Schlafly & Finkbeiner (2011)
EBV_MW = {'IC5332': 0.015,'NGC0628': 0.062,'NGC1087': 0.03,'NGC1300': 0.026,
          'NGC1365': 0.018,'NGC1385': 0.018,'NGC1433': 0.008,'NGC1512': 0.009,
          'NGC1566': 0.008,'NGC1672': 0.021,'NGC2835': 0.089,'NGC3351': 0.024,
          'NGC3627': 0.037,'NGC4254': 0.035,'NGC4303': 0.02,'NGC4321': 0.023,
          'NGC4535': 0.017,'NGC5068': 0.091,'NGC7496': 0.008}

sample = ['IC5332','NGC0628','NGC1087','NGC1300','NGC1365','NGC1385',
          'NGC1433','NGC1512','NGC1566','NGC1672','NGC2835','NGC3351',
          'NGC3627','NGC4254','NGC4303','NGC4321','NGC4535','NGC5068',
          'NGC7496']

sigclip = SigmaClip(sigma=3.0, maxiters=10)

for gal_name in tqdm(sample):

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
    # read the compact cluster and DOLPHOT catalogues
    cluster_filename = data_ext/'Products'/'compact_clusters'/f'PHANGS_IR3_{gal_name.lower()}_phangs-hst_v1p1_human_class12.fits'
    if not cluster_filename.is_file():
        print(f'no compact clusters for {gal_name}')
        continue
    else:     
        with fits.open(cluster_filename) as hdul:
            compact_clusters = Table(hdul[1].data)
        compact_clusters['SkyCoord'] = SkyCoord(compact_clusters['PHANGS_RA']*u.deg,compact_clusters['PHANGS_DEC']*u.deg)


    # the output table with columns for the Halpha/Hbeta fluxes
    tbl = compact_clusters[['INDEX','ID_PHANGS_CLUSTERS','PHANGS_X','PHANGS_Y','PHANGS_RA','PHANGS_DEC','SkyCoord']].copy()
    tbl.add_column(gal_name,index=0,name='gal_name')
    tbl['in_frame'] = ~np.isnan(get_value(Halpha.data,np.array(tbl['SkyCoord'].to_pixel(Halpha.wcs)).T.astype(int)))

    tbl['HA6562_FLUX'] = np.nan
    tbl['HA6562_FLUX_BKG'] = np.nan
    tbl['HB4861_FLUX'] = np.nan
    tbl['HB4861_FLUX_BKG'] = np.nan
    tbl['HA6562_FLUX_ERR'] = np.nan
    tbl['HB4861_FLUX_ERR'] = np.nan

    aperture = SkyCircularAperture(tbl['SkyCoord'],r=r)
    annulus_aperture = SkyCircularAnnulus(tbl['SkyCoord'],r_in=r_in,r_out=r_out)


    bkg_stats = ApertureStats(data=Halpha.data,aperture=annulus_aperture,
                              sigma_clip=sigclip,wcs=Halpha.wcs)
    # just ignore NaNs for the time beeing (the aperture sum will be 0 anyways)
    bkg_median = bkg_stats.median
    bkg_median[np.isnan(bkg_median)] = 0
                            
    aper_stats = ApertureStats(data=Halpha.data,aperture=aperture,
                               error=Halpha.uncertainty.array,
                               wcs=Halpha.wcs,local_bkg=bkg_median)
    tbl['HA6562_FLUX'] = aper_stats.sum
    tbl['HA6562_FLUX_ERR'] = aper_stats.sum_err
    tbl['HA6562_FLUX_BKG'] = aper_stats.sum_aper_area.value*bkg_median

    # do the same for Hbeta
    bkg_stats = ApertureStats(data=Hbeta.data,aperture=annulus_aperture,
                              sigma_clip=sigclip,wcs=Hbeta.wcs)
    # just ignore NaNs for the time beeing (the aperture sum will be 0 anyways)
    bkg_median = bkg_stats.median
    bkg_median[np.isnan(bkg_median)] = 0
                            
    aper_stats = ApertureStats(data=Hbeta.data,aperture=aperture,
                               error=Hbeta.uncertainty.array,
                               wcs=Hbeta.wcs,local_bkg=bkg_median)
    tbl['HB4861_FLUX'] = aper_stats.sum
    tbl['HB4861_FLUX_ERR'] = aper_stats.sum_err
    tbl['HB4861_FLUX_BKG'] = aper_stats.sum_aper_area.value*bkg_median

    # Milky Way extinction
    rc_MW = pyneb.RedCorr(E_BV = EBV_MW[gal_name], R_V = 3.1, law = 'CCM89 oD94')

    tbl['HA6562_FLUX'] *= rc_MW.getCorr(6562) 
    tbl['HB4861_FLUX'] *= rc_MW.getCorr(4861)
    tbl['HA6562_FLUX_BKG'] *= rc_MW.getCorr(6562) 
    tbl['HB4861_FLUX_BKG'] *= rc_MW.getCorr(4861)
    tbl['HA6562_FLUX_ERR'] *= rc_MW.getCorr(6562) 
    tbl['HB4861_FLUX_ERR'] *= rc_MW.getCorr(4861)

    # Internal extinction is estimated from the Balmer decrement
    rc_balmer = pyneb.RedCorr(R_V = 3.1, law = 'CCM89 oD94')
    rc_balmer.setCorr(obs_over_theo= tbl['HA6562_FLUX']/tbl['HB4861_FLUX'] / 2.86, wave1=6562.81, wave2=4861.33)
    # set E(B-V) to zero if S/N is less than 3 in Halpha or Hbeta
    rc_balmer.E_BV[(rc_balmer.E_BV<0) | (tbl['HB4861_FLUX']<3*tbl['HB4861_FLUX_ERR']) |  (tbl['HA6562_FLUX']<3*tbl['HA6562_FLUX_ERR'])] = 0
    #tbl['A5007'] =  -2.5*np.log10(rc.getCorr(5007))

    tbl['HA6562_FLUX_CORR'] = tbl['HA6562_FLUX'] * rc_balmer.getCorr(6562) 
    tbl['HB4861_FLUX_CORR'] = tbl['HB4861_FLUX'] * rc_balmer.getCorr(4861)
    tbl['HA6562_FLUX_CORR_ERR'] = tbl['HA6562_FLUX_ERR'] * rc_balmer.getCorr(6562) 
    tbl['HB4861_FLUX_CORR_ERR'] = tbl['HB4861_FLUX_ERR'] * rc_balmer.getCorr(4861)
    tbl['EBV_balmer'] = rc_balmer.E_BV

    del tbl['SkyCoord']

    print('write to file')
    primary_hdu = fits.PrimaryHDU()
    for i,comment in enumerate(doc.split('\n')):
        if i==0:
            primary_hdu.header['COMMENT'] = comment
        else:
            primary_hdu.header[''] = comment
    table_hdu   = fits.BinTableHDU(tbl)
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(out_folder / (cluster_filename.stem+'_Halpha.fits'),overwrite=True)
