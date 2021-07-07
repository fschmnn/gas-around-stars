from skimage.measure import find_contours
from regions import PixCoord, PolygonPixelRegion
import numpy as np 

def find_sky_region(mask,wcs):
    '''create a region object from a mask and wcs
    
    Returns:
    reg_pix : PixelRegion
    reg_sky : SkyRegion
    '''

    mask[:,[0,-1]]=1
    mask[[0,-1],:]=1

    # find the contours around the image to use as vertices for the PixelRegion
    contours = find_contours(mask,0.5,)
    # we use the region with the most vertices
    coords = max(contours,key=len)
    #coords = np.concatenate(contours)

    # the coordinates from find_counters are switched compared to astropy
    reg_pix  = PolygonPixelRegion(vertices = PixCoord(*coords.T[::-1])) 
    reg_sky  = reg_pix.to_sky(wcs)
    
    return reg_pix, reg_sky


# check for bordering regions
from skimage.segmentation import find_boundaries
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt

def find_neighbors(mask,position,label,size=16,plot=False):
    cutout = Cutout2D(mask,position,size=(size,size),mode='partial',fill_value=np.nan)
    boundaries = find_boundaries(cutout.data==label,mode='outer')
    
    if plot:
        fig,ax=plt.subplots(figsize=(5,5))
        ax.imshow(cutout.data,origin='lower')
        mask = np.zeros((*cutout.shape,4))
        mask[boundaries & ~np.isnan(cutout.data),:] = (1,0,0,1)
        ax.imshow(mask,origin='lower')
    return set(cutout.data[boundaries][~np.isnan(cutout.data[boundaries])].astype(int))
