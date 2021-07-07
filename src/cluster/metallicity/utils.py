import numpy as np

def diagnostic_line_ratios(table,append=True):
    '''calculate diagnostic line ratios
    
    They are commonly used in metallicity prescriptions

    used in Pilyugin+2016 and Perez-Montero+2017
    '''

    with np.errstate(divide='ignore'):
        # used in strong line methods
        table['N2']  = 4/3*table['NII6583_FLUX_CORR'] / table['HB4861_FLUX_CORR']
        table['R2']  = 2.4*table['OII3726_FLUX_CORR'] / table['HB4861_FLUX_CORR']
        table['R3']  = 4/3*table['OIII5006_FLUX_CORR']     / table['HB4861_FLUX_CORR']
        table['S2']  = (table['SII6716_FLUX_CORR']+table['SII6730_FLUX_CORR']) / table['HB4861_FLUX_CORR']
        table['R23'] = table['R2'] + table['R3']

        # used in direct methods
        table['RS2'] = table['SII6716_FLUX_CORR'] / table['SII6730_FLUX_CORR']
        table['RO2'] = 2.4*table['OII3726_FLUX_CORR']/(table['OII7319_FLUX_CORR']+table['OII7330_FLUX_CORR'])
        table['RS3'] = (3.44*table['SIII9068_FLUX_CORR'])/table['SIII6312_FLUX_CORR']
        table['RN2'] = 4/3*table['NII6583_FLUX_CORR']/table['NII5754_FLUX_CORR']

    if append==True:
        return table
    else:
        return table[['N2','R2','R3','S2','R23','RS2','RO2','RS3','RN2']]