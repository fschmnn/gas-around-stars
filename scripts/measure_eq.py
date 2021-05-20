from pathlib import Path
import logging
import sys
from tqdm import tqdm

import numpy as np

from astropy.io import ascii, fits
from astropy.table import Table 
import astropy.units as u 
import astropy.constants as c
from astropy.coordinates import Distance

import matplotlib.pyplot as plt

from cluster.spectrum import fit_emission_line


basedir = Path('..')
data_ext = Path('a:')


sample_table = ascii.read(basedir/'..'/'pnlf'/'data'/'interim'/'sample.txt')
sample_table.add_index('name')


# nebulae catalogue from Francesco (mostly HII-regions)
with fits.open(basedir/'data'/'interim'/'Nebulae_Catalogue_with_FUV_v2p1.fits') as hdul:
    nebulae = Table(hdul[1].data)
nebulae['eq_width'] = np.nan


for name in tqdm(np.unique(nebulae['gal_name']),position=0,leave=False,colour='green'):
    
    p = {x:sample_table.loc[name][x] for x in sample_table.columns}

    filename = data_ext / 'MUSE_DR2.1' / 'Nebulae catalogue' /'spectra'/f'{name}_VorSpectra.fits'
    with fits.open(filename) as hdul:
        spectra = Table(hdul[1].data)
        spectral_axis = np.exp(Table(hdul[2].data)['LOGLAM'])*u.Angstrom
        
    spectra['region_ID'] = np.arange(len(spectra))
    spectra.add_index('region_ID')

    H0 = 67 * u.km / u.s / u.Mpc
    z = (H0*Distance(distmod=p['(m-M)'])/c.c).decompose()
    lam_HA0 = 6562.8*u.Angstrom
    lam_HA = (1+z)*lam_HA0
    
    sub = nebulae[nebulae['gal_name']==name]

    for row in tqdm(sub,position=1,leave=False,colour='red',desc=name):
        try:
            region_ID = row['region_ID']
            flux = spectra.loc[region_ID]['SPEC']*u.erg/u.s/u.cm**2/u.A
            fit = fit_emission_line(spectral_axis,flux,lam_HA)
            integrated_flux = fit.amplitude_0*np.sqrt(np.pi)*np.exp(-1/(2*fit.stddev_0**2)) * u.erg/u.s/u.cm**2
            continuum = fit.c0_1 * u.erg/u.s/u.cm**2/u.Angstrom
            #eq_width = integrated_flux/continuum
            eq_width = row['HA6562_FLUX']/continuum
            row['eq_width'] = eq_width.value
        except:
            print(f'error for {name} {region_ID}')

# write to file
primary_hdu = fits.PrimaryHDU()
table_hdu   = fits.BinTableHDU(nebulae)
hdul = fits.HDUList([primary_hdu, table_hdu])
hdul.writeto(basedir/'data'/'interim'/'Nebulae_Catalogue_with_FUV_eq_v2p1.fits',overwrite=True)