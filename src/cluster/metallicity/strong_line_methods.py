"""
Strong Line method ("empirical method)
======================================

from Pilyugin+2016

* R calibration
* S calibration
"""

import numpy as np 

def strong_line_metallicity_R(R2,R3,N2):
    '''calculate 12+log(O/H) with R calibration
    
    based on  Pilyugin+2016
    '''
    
    mask = np.log10(N2)>-0.6
    OH = np.zeros_like(R2)
    print(f'upper branch: {np.sum(mask)}, lower branch: {np.sum(~mask)}')
    
    with np.errstate(divide='ignore'):
        # first upper branch
        a1,a2,a3,a4,a5,a6=8.589,0.022,0.399,-0.137,0.164,0.589
        OH[mask] = a1+a2*np.log10(R3[mask]/R2[mask])+a3*np.log10(N2[mask])+(a4+a5*np.log10(R3[mask]/R2[mask])+a6*np.log10(N2[mask]))*np.log10(R2[mask])

        # now lower branch
        a1,a2,a3,a4,a5,a6=7.932,0.944,0.695,0.970,-0.291,-0.019
        OH[~mask] = a1+a2*np.log10(R3[~mask]/R2[~mask])+a3*np.log10(N2[~mask])+(a4+a5*np.log10(R3[~mask]/R2[~mask])+a6*np.log10(N2[~mask]))*np.log10(R2[~mask])

    return OH

def strong_line_metallicity_S(S2,R3,N2):
    '''calculate 12+log(O/H) with S calibration
    
    based on  Pilyugin+2016
    '''
    
    mask = np.log10(N2)>-0.6
    OH = np.zeros_like(S2)
    print(f'upper branch: {np.sum(mask)}, lower branch: {np.sum(~mask)}')
    
    with np.errstate(divide='ignore'):
        # first upper branch
        a1,a2,a3,a4,a5,a6=8.424,0.030,0.751,-0.349,0.182,0.508
        OH[mask] = a1+a2*np.log10(R3[mask]/S2[mask])+a3*np.log10(N2[mask])+(a4+a5*np.log10(R3[mask]/S2[mask])+a6*np.log10(N2[mask]))*np.log10(S2[mask])

        # now lower branch
        a1,a2,a3,a4,a5,a6=8.072,0.789,0.726,1.069,-0.170,0.022
        OH[~mask] = a1+a2*np.log10(R3[~mask]/S2[~mask])+a3*np.log10(N2[~mask])+(a4+a5*np.log10(R3[~mask]/S2[~mask])+a6*np.log10(N2[~mask]))*np.log10(S2[~mask])

    return OH