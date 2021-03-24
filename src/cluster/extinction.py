import numpy as np
import matplotlib.pyplot as plt
from dust_extinction.parameter_averages import O94, CCM89

extinction_model = CCM89(Rv=3.1)

def extinction(EBV,EBV_err,wavelength,plot=False):
    '''Calculate the extinction for a given EBV and wavelength with errors
    
    
    flux_corr = flux / ext
    
    '''
    
    EBV = np.atleast_1d(EBV)
    sample_size = 100000

    ext = extinction_model.extinguish(wavelength,Ebv=EBV)
    
    EBV_rand = np.random.normal(loc=EBV,scale=EBV_err,size=(sample_size,len(EBV)))
    ext_arr  = extinction_model.extinguish(wavelength,Ebv=EBV_rand)
        
    ext_err  = np.std(ext_arr,axis=0)
    ext_mean = np.mean(ext_arr,axis=0)
    
    if plot:
        fig,(ax1,ax2) =plt.subplots(nrows=1,ncols=2,figsize=(two_column,two_column/2))
        ax1.hist(EBV_rand[:,0],bins=100)
        ax1.axvline(EBV[0],color='black')
        ax1.set(xlabel='E(B-V)')
        ax2.hist(ext_arr[:,0],bins=100)
        ax2.axvline(ext[0],color='black')
        ax2.set(xlabel='extinction')
        plt.show()
 
    return ext,ext_err