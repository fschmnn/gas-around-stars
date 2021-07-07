from pathlib import Path
from astropy.io import fits 
from astropy.table import Table, vstack

basedir  = Path('..')
data_ext = Path('a:')

scalepc = 64
HSTband = 'nuv'
version = 'v1p1'

folder=data_ext/'HST'/'associations v1p1'/'multi-scale stellar associations'

lst = []
for folder in [x for x in folder.iterdir() if x.stem[-1].isdigit()]:
    gal_name = folder.stem
    folder = folder/f'{gal_name}_{HSTband}_tracerstars'/f'{scalepc}pc'
    if not folder.is_dir():
        print(f'no catalogue for {gal_name} at {scalepc}pc')
        continue 
        
    catalogue_file = folder / f'{gal_name}_phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits'
    with fits.open(catalogue_file) as hdul:
        tbl = Table(hdul[1].data)
    tbl.add_column(gal_name.upper(),name='gal_name',index=0)
    tbl.rename_column('reg_id','assoc_ID')
    lst.append(tbl)
    
print(f'{len(lst)} galaxies for resolution {scalepc}pc')
associations = vstack(lst)

hdu = fits.BinTableHDU(associations,name='associations')
hdu.writeto(basedir/'data'/'interim'/f'phangshst_associations_{HSTband}_ws{scalepc}pc_{version}.fits',overwrite=True)