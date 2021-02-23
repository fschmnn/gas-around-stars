def extract_spectra(cube,header,mask,regions,filename):
    '''extract spectra from a spectral cube at given positions'''
    
    logger.info(f'extracting spectrum for {len(regions)} objects')
    
    wavelength = []
    spectrum   = []
    
    for i,region_ID in enumerate(regions):
        print(f'{i+1} of {len(regions)}')

        spectrum.append(np.sum(data_cube[...,mask==region_ID],axis=1))  
        wavelength.append(np.linspace(header['CRVAL3'],header['CRVAL3']+header['NAXIS3']*header['CD3_3'],header['NAXIS3']))
    
    spectra = Table(data=[regions,wavelength,spectrum],
                    names=['region_ID','wavelength','spectrum'])
    
    return spectra

    
filename = basedir/'data'/'interim'/f'{name}_nebulae_spectra.fits'
spectra = extract_spectra(data_cube,cube_header,nebulae_mask.data,isolated_nebulae[:10],filename)
spectra.add_index('region_ID')

hdu = fits.BinTableHDU(spectra,name='spectra')
hdu.writeto(filename,overwrite=True)