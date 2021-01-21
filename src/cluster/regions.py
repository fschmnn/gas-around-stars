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