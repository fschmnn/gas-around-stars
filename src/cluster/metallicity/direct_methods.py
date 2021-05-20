"""
Auroral line methods ("Direct methods")
=======================================

Calculate electron temperature, density and abundances from auroral lines

based on Perez-Montero+2017
"""

import numpy as np


def electron_density_sulfur(RS2,te):
    '''calculate electron density from [SII]
    
    based on Perez-Montero+2017 (formula 15,16,17)

    Parameters
    ----------
    RS2 : [SII]6716/[SII]6731

    t : temperature in units of 10^4 K (usually from [OIII])
    
    Returns
    -------
    electron density in units of cm^-3
    '''

    a0 = 16.054-7.79/te-11.32*te
    a1 = -22.6+11.08/te+16.02*te
    b0 = -21.61+11.89/te+14.59*te
    b1 = 9.17-5.09/te-6.18*te
    
    return 1e3 * (RS2*a0+a1)/(RS2*b0+b1)
    

def electron_temperature_oxygen(R02,ne):
    '''calculate electron temperature from [OII]
    
    based on Perez-Montero+2017 (formula 9,11,12)

    Parameters
    ----------
    R02 : ([OII]3727+[OII]3729) / ([OII]3719+[OII]3730)

    ne : electron density in units of cm^-3

    Returns
    -------
    electron temperature in units of 10^4 K    
    '''
    
    a0 = 0.2526-0.000357*ne-0.43/ne
    a1 = 0.00136+0.00481/ne
    a2 = 35.624-0.0172*ne-25.12/ne

    return a0+a1*R02+a2/R02


def electron_temperature_sulfur(RS3):
    '''calculate electron temperature from [SIII]
    
    based on Perez-Montero+2017 (formula 22,23)

    Parameters
    ----------
    RS3 : ([SIII]9069+[SIII]9532) / [SIII]6312

    Returns
    -------
    electron temperature in units of 10^4 K    
    '''

    return 0.5147+0.0003187*RS3+23.64041/RS3


def electron_temperature_nitrogen(RN2):
    '''calculate electron temperature from [NII]
    
    based on Perez-Montero+2017 (formula 25,26)
    
    theoretical ratio: [NII]6583 = 2.9 [NII]6548

    Note 
    Parameters
    ----------
    RN2 : ([NII]6548+[NII]6583)  [NII]5755

    Returns
    -------
    electron temperature in units of 10^4 K    
    '''

    return 0.6153-0.0001529*RN2+35.3641/RN2 


def oxygen_abundance_direct(R2,R3,te,ne):
    '''calculate 12+log(O+/H) with direct method

    based on Perez-Montero+2017 (formula 38,40,41)

    Parameters
    ----------
    R2 : ([OII]3727+[OII]3729) / HB4861

    R3 : ([OIII]4959+[OII]5007) / HB4861

    te : electron temperature in units of 1e4 K

    ne : electron density in units of cm^-3

    Returns 
    -------
    12+log(O/H) from direct method
    '''

    #OI = np.log10(OII7320)+7.21+2.511/te-0.422*np.log10(te)+1e-3.4*ne*(1-1e-3.44*ne)

    OI  = np.log10(R2)+5.887+1.641/te-0.543*np.log10(te)+0.000114*ne
    OII = np.log10(R3)+6.1868+1.2491/te-0.5816*np.log10(te)
    
    #12+np.log10(10**(OI-12)+10**(OII-12))
    return np.log10(10**(OI)+10**(OII))
    

