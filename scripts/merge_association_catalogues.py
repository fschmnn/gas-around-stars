from pathlib import Path
from astropy.io import fits 
from astropy.table import Table, vstack
import astropy.units as u
from astropy.units import spectral_density

basedir  = Path('..')
data_ext = Path('a:')/'Archive'

version = 'v1p2'

band_wavelength = {'NUV':270.4*u.AA,'U':335.5*u.AA,'B':432.5*u.AA,'V':530.8*u.AA,'I':802.4*u.AA}
freq_to_wave = lambda band: u.mJy.to(u.erg/u.s/u.cm**2/u.Angstrom,equivalencies=spectral_density(band_wavelength[band]))

base_folder=data_ext/'Products'/'stellar_associations'/f'associations_{version}'

for scalepc in [8,16,32,64]:
    for HSTband in ['nuv','v']:
        print(f'working on {scalepc}pc {HSTband}')

        lst = []
        for folder in [x for x in base_folder.iterdir() if x.stem.endswith(f'_{HSTband}')]:
            gal_name = folder.stem.split('_')[0]
            folder = folder/f'{scalepc}pc'
            if not folder.is_dir():
                print(f'no catalogue for {gal_name} at {scalepc}pc {HSTband}')
                continue 
                
            catalogue_file = folder / f'{gal_name}_phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits'
            with fits.open(catalogue_file) as hdul:
                tbl = Table(hdul[1].data)
            tbl.add_column(gal_name.upper(),name='gal_name',index=0)
            tbl.rename_column('reg_id','assoc_ID')
            lst.append(tbl)
            
        print(f'{len(lst)} galaxies for resolution {scalepc}pc')
        associations = vstack(lst)

        # add the flux density in erg s-1 cm-2 AA-1
        for col in list(associations.columns):
            if col.endswith('mjy'):
                band = col.split("_")[0]
                associations[f'{band}_FLUX'] = associations[col]*freq_to_wave(band)
                associations[f'{band}_FLUX_ERR'] = associations[col+'_err']*freq_to_wave(band)

        hdu = fits.BinTableHDU(associations,name='associations')
        hdu.writeto(basedir/'data'/'interim'/f'phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits',overwrite=True)