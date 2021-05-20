from astropy.modeling import models, fitting
import numpy as np 
import astropy.units as u 
import matplotlib.pyplot as plt 

single_column = 3.321 # in inch
two_column    = 6.974 # in inch

def fit_emission_line(spectral_axis,flux,lam0,filename=None):
    '''fit a gaussian + a constant background to a spectrum
    
    Parameters
    ----------
    
    spectra_axis : numpy array (with astropy units)
    
    flux : numpy array (with astropy units)
    
    lam0 : astropy Quantity
        the wavelength of the line 
    '''
    
    # Halpha should be the brightest peak in this range
    #print(f'guess={lam0:.1f}')
    idx = np.argmax(flux[(spectral_axis>lam0-50*u.Angstrom) & (spectral_axis<lam0+50*u.Angstrom) ])
    lam0 = spectral_axis[(spectral_axis>lam0-50*u.Angstrom) & (spectral_axis<lam0+50*u.Angstrom) ][idx]
    #print(f'peak={lam0:.1f}')

    region = (spectral_axis>lam0-10*u.Angstrom) & (spectral_axis<lam0+15*u.Angstrom) 
    amplitude_guess = np.max(flux[region].value)
    
    channel_width = np.mean(np.ediff1d(spectral_axis[region]))
    
    # create a model with a gaussian + a constant 
    model = models.Gaussian1D(amplitude_guess, lam0.value, 1) + models.Polynomial1D(degree=0)
    model.amplitude_0.min = 0
    model.mean_0.bounds = (lam0.value-3,lam0.value+3)
    fitter = fitting.LevMarLSQFitter()
    fit = fitter(model, spectral_axis[region].value, flux[region].value)
    
    integrated_flux = fit.amplitude_0*np.sqrt(np.pi)*np.exp(-1/(2*fit.stddev_0**2)) * u.erg/u.s/u.cm**2
    continuum = fit.c0_1 * u.erg/u.s/u.cm**2/u.Angstrom
    eq_width = channel_width.value*integrated_flux/continuum
    
    if filename:
        f, ax = plt.subplots(figsize=(single_column,single_column/1.618))  
        ax.plot(spectral_axis.value, flux.value,color='black') 
        ax.plot(spectral_axis[region].value,fit(spectral_axis[region].value),color='tab:red')
        ax.plot(spectral_axis.value,0*spectral_axis.value+fit.c0_1,ls='--',color='tab:red')
        ax.axvline(fit.mean_0.value,color='black',ls='--')
        ax.fill_between(spectral_axis,fit(spectral_axis.value),0*spectral_axis.value+fit.c0_1,alpha=0.5)
        
        eq_region = (spectral_axis>lam0+7*u.Angstrom) & (spectral_axis<lam0+7*u.Angstrom+eq_width) 
        ax.fill_between(spectral_axis[eq_region],0*spectral_axis[eq_region].value+fit.c0_1,0*spectral_axis[eq_region].value,alpha=0.5)
        
        ax.set(xlim=(lam0-30*u.Angstrom, lam0+30*u.Angstrom),ylabel='flux',xlabel=r'wavelength / $\AA$')
        plt.savefig(filename,dpi=300)
        plt.show()
        
    return fit
