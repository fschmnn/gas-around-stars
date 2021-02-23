import numpy as np
from scipy.stats import binned_statistic


def bin_stat(x,y,xlim,nbins=10,statistic='mean'):
    '''calculate the binned statistics'''

    # just ignore nan values
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    mean, edges, _ = binned_statistic(x,y,statistic=statistic,bins=nbins,range=xlim)
    std, _, _ = binned_statistic(x,y,statistic='std',bins=nbins,range=xlim)
    return (edges[1:]+edges[:-1])/2,mean,std