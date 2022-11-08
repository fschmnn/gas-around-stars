import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u


from reproject import reproject_interp
from regions import PixCoord, RectangleSkyRegion
from skimage.measure import find_contours

from pnlf.plot import add_scale, create_RGB

single_column = 3.321 # in inch
two_column    = 6.974 # in inch

def single_cutout(ax,position,image,mask1=None,mask2=None,points=None,label=None,size=6*u.arcsec):
    position = position.directional_offset_by(0*u.deg,0.3*u.arcsec)

    cutout_image = Cutout2D(image.data,position,size=size,wcs=image.wcs)
    norm = simple_norm(cutout_image.data,clip=False,stretch='linear',percent=99.5)
    ax.imshow(cutout_image.data,origin='lower',norm=norm,cmap=plt.cm.gray_r)

    # plot the nebulae catalogue
    cutout_mask, _  = reproject_interp(mask1,output_projection=cutout_image.wcs,shape_out=cutout_image.shape,order='nearest-neighbor')    
    region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

    contours = []
    for i in region_ID:
        blank_mask = np.zeros_like(cutout_mask)
        blank_mask[cutout_mask==i] = 1
        contours += find_contours(blank_mask, 0.5)

    for coords in contours:
        ax.plot(coords[:,1],coords[:,0],color='tab:red',lw=1,label=r'H\textsc{ii} region')


    mask = np.zeros((*cutout_mask.shape,4))
    mask[~np.isnan(cutout_mask.data),:] = (0.84, 0.15, 0.16,0.1)
    ax.imshow(mask,origin='lower',cmap=plt.cm.gist_heat)

    # plot the association catalogue
    if mask2:
        cutout_mask, _  = reproject_interp(mask2,output_projection=cutout_image.wcs,shape_out=cutout_image.shape,order='nearest-neighbor')    
        region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

        contours = []
        for i in region_ID:
            blank_mask = np.zeros_like(cutout_mask)
            blank_mask[cutout_mask==i] = 1
            contours += find_contours(blank_mask, 0.5)

        for coords in contours:
            ax.plot(coords[:,1],coords[:,0],color='tab:blue',lw=1,label='association')

        mask = np.zeros((*cutout_mask.shape,4))
        mask[~np.isnan(cutout_mask.data),:] = (0.12,0.47,0.71,0.1)
        ax.imshow(mask,origin='lower')

    # mark the position of the clusters within the cutout
    if points:
        
        region = RectangleSkyRegion(position,0.9*size,0.9*size)
        in_frame = points[region.contains(points['SkyCoord'],cutout_image.wcs)]
        for row in in_frame:
            x,y = row['SkyCoord'].to_pixel(cutout_image.wcs)
            if 5<x<cutout_image.data.shape[0]-5 and 5<y<cutout_image.data.shape[1]-5:
                ax.scatter(x,y,marker='o',facecolors='none',s=40,lw=0.8,color='green',label='DOLPHOT peaks')
                if 'label' in points.columns:
                    ax.annotate(row['label'], (x+4, y),color='green')
 
    if label:
        t = ax.text(0.0663,0.8666,label, transform=ax.transAxes,color='black',fontsize=7)
        t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))

    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def single_cutout_rgb(ax,position,r,g,b,mask1=None,mask2=None,points=None,label=None,size=6*u.arcsec):
    
    cutout_b = Cutout2D(b.data,position,size=size,wcs=b.wcs)
    cutout_r, _  = reproject_interp(r,output_projection=cutout_b.wcs,shape_out=cutout_b.shape)    
    cutout_g, _  = reproject_interp(g,output_projection=cutout_b.wcs,shape_out=cutout_b.shape)    

    rgb = create_RGB(cutout_r,cutout_g,cutout_b.data,
                        percentile=[98,98,98],weights=[0.81,.81,.81])

    ax.imshow(rgb,origin='lower')

    # plot the nebulae catalogue
    cutout_mask, _  = reproject_interp(mask1,output_projection=cutout_b.wcs,shape_out=cutout_b.shape,order='nearest-neighbor')    
    region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

    contours = []
    for i in region_ID:
        blank_mask = np.zeros_like(cutout_mask)
        blank_mask[cutout_mask==i] = 1
        contours += find_contours(blank_mask, 0.5)

    for coords in contours:
        ax.plot(coords[:,1],coords[:,0],color='tab:red',lw=1,label=r'H\textit{ii}-region')


    mask = np.zeros((*cutout_mask.shape,4))
    mask[~np.isnan(cutout_mask.data),:] = (0.84, 0.15, 0.16,0.1)
    ax.imshow(mask,origin='lower')

    # plot the association catalogue
    if mask2:
        cutout_mask, _  = reproject_interp(mask2,output_projection=cutout_b.wcs,shape_out=cutout_b.shape,order='nearest-neighbor')    
        region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

        contours = []
        for i in region_ID:
            blank_mask = np.zeros_like(cutout_mask)
            blank_mask[cutout_mask==i] = 1
            contours += find_contours(blank_mask, 0.5)

        for coords in contours:
            ax.plot(coords[:,1],coords[:,0],color='tab:blue',lw=1,label='association')

        mask = np.zeros((*cutout_mask.shape,4))
        mask[~np.isnan(cutout_mask.data),:] = (0.12,0.47,0.71,0.1)
        ax.imshow(mask,origin='lower')

    # mark the position of the clusters within the cutout
    if points:
        region = RectangleSkyRegion(position,0.9*size,0.9*size)
        in_frame = points[region.contains(points['SkyCoord'],cutout_b.wcs)]
        for row in in_frame:
            x,y = row['SkyCoord'].to_pixel(cutout_b.wcs)
            if 5<x<cutout_b.data.shape[0]-5 and 5<y<cutout_b.data.shape[1]-5:
                ax.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.4,color='tab:blue',label='compact cluster')

    if label:
        t = ax.text(0.07,0.875,label, transform=ax.transAxes,color='black',fontsize=7)
        t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))

    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

import astrotools.plot.multicolorfits as mcf

def create_multicolorimage(*args):
    '''combine multiple images to a three color image'''
    
    color_images = []
    for arg in args:
        color = arg.pop('color')
        greyscale = mcf.greyRGBize_image(**arg)
        color_images.append(mcf.colorize_image(greyscale,color, colorintype='hex',gammacorr_color=2.2))
    
    return mcf.combine_multicolor(color_images, gamma=2.2)


def single_cutout_hst(ax,position,images,mask1=None,mask2=None,points=None,label=None,size=6*u.arcsec):
    position = position.directional_offset_by(0*u.deg,0.3*u.arcsec)

    f275w_cutout  = Cutout2D(images.f275w.data,position,size=size,wcs=images.f275w.wcs)
    f336w_cutout = reproject_interp(images.f336w,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,return_footprint=False)    
    f438w_cutout = reproject_interp(images.f438w,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,return_footprint=False)    
    f555w_cutout = reproject_interp(images.f555w,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,return_footprint=False)    
    f814w_cutout = reproject_interp(images.f814w,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,return_footprint=False)    
    halpha_cutout  = reproject_interp(images.Halpha,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,return_footprint=False)    
    halpha_cutout_muse  = reproject_interp(images.Halpha,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,return_footprint=False)    

    p1,p2 = np.nanpercentile(halpha_cutout_muse.data,[0,100])
    f275_white  = {'color':'#d5c5e3','datin':f275w_cutout.data,'rescalefn':'linear','scaletype':'perc','min_max':[15,99.9]}
    f336_purple = {'color':'#9C4FFF','datin':f336w_cutout.data,'rescalefn':'linear','scaletype':'perc','min_max':[15,99.9]}
    f438_blue   = {'color':'#61b3cf','datin':f438w_cutout.data,'rescalefn':'linear','scaletype':'perc','min_max':[15,99.8]}
    f555_green  = {'color':'#00e32d','datin':f555w_cutout.data,'rescalefn':'linear','scaletype':'perc','min_max':[15,99.8]}
    f814_yellow = {'color':'#e08e1b','datin':f814w_cutout.data,'rescalefn':'linear','scaletype':'perc','min_max':[15,99.8]}
    ha_red_peak = {'color':'#ed7f5a','datin':halpha_cutout_muse.data,'rescalefn':'linear','scaletype':'perc','min_max':[70,99.8]}
    ha_red_dif  = {'color':'#ed5a95','datin':halpha_cutout_muse.data,'rescalefn':'log','scaletype':'abs','min_max':[4*p1,8*p2]}


    rgb = create_multicolorimage(f275_white,f336_purple,f438_blue,f555_green,f814_yellow,ha_red_peak,ha_red_dif)

    ax.imshow(rgb,origin='lower')

    # plot the nebulae catalogue
    cutout_mask, _  = reproject_interp(mask1,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,order='nearest-neighbor')    
    region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

    contours = []
    for i in region_ID:
        blank_mask = np.zeros_like(cutout_mask)
        blank_mask[cutout_mask==i] = 1
        contours += find_contours(blank_mask, 0.5)

    for coords in contours:
        ax.plot(coords[:,1],coords[:,0],color='tab:red',lw=1,label=r'H\textsc{ii} region')


    mask = np.zeros((*cutout_mask.shape,4))
    mask[~np.isnan(cutout_mask.data),:] = (0.84, 0.15, 0.16,0.1)
    ax.imshow(mask,origin='lower')

    # plot the association catalogue
    if mask2:
        cutout_mask, _  = reproject_interp(mask2,output_projection=f275w_cutout.wcs,shape_out=f275w_cutout.shape,order='nearest-neighbor')    
        region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

        contours = []
        for i in region_ID:
            blank_mask = np.zeros_like(cutout_mask)
            blank_mask[cutout_mask==i] = 1
            contours += find_contours(blank_mask, 0.5)

        for coords in contours:
            ax.plot(coords[:,1],coords[:,0],color='tab:blue',lw=1,label='association')

        mask = np.zeros((*cutout_mask.shape,4))
        mask[~np.isnan(cutout_mask.data),:] = (0.12,0.47,0.71,0.1)
        ax.imshow(mask,origin='lower')

    # mark the position of the clusters within the cutout
    if points:
        region = RectangleSkyRegion(position,0.9*size,0.9*size)
        in_frame = points[region.contains(points['SkyCoord'],f275w_cutout.wcs)]
        for row in in_frame:
            x,y = row['SkyCoord'].to_pixel(f275w_cutout.wcs)
            if 5<x<f275w_cutout.data.shape[0]-5 and 5<y<f275w_cutout.data.shape[1]-5:
                ax.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.4,color='tab:blue',label='compact cluster')

    if label:
        t = ax.text(0.068,0.868,label, transform=ax.transAxes,color='black',fontsize=7)
        t.set_bbox(dict(facecolor='white', alpha=0.8, ec='white',lw=0))

    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def multi_cutout(positions,image,mask1=None,mask2=None,points=None,labels=None,
                 width = two_column,filename=None,size=6*u.arcsec,ncols=4):
    '''Plot multiple cutouts with the position of the clusters
    
    Parameters
    ----------

    image : NDData
        the background image that is used
    positions : SkyCoord
        the position of the cutouts
    masks : NDData
        A mask with outlines
    points : SkyCoord
        Points to mark in the image

    '''
    
    ncols = ncols
    nrows = int(np.ceil(len(positions)/ncols))

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
    axes_iter = iter(axes.flatten())

    for position,label in zip(positions,labels):  

        ax = next(axes_iter)
        ax = single_cutout(ax,
                            position = position,
                            image = image,
                            mask1 = mask1,
                            mask2 = mask2,
                            label = label,
                            size  = 4*u.arcsecond)

    h,l = fig.axes[0].get_legend_handles_labels()
    ax = next(axes_iter)
    ax.axis('off')
    ax.legend(h[::len(h)-1],l[::(len(l)-1)],fontsize=7,loc='upper center',frameon=False)

    if True:
        length = 0.263
        ax.set(xlim=[0,1],ylim=[0,1])
        x,y = 0.2,0.2
        ax.plot([x,x+length],[y,y],color='black',marker='|',lw=1,ms=4)
        ax.text(x+0.5*length,y*1.3,'100 pc',horizontalalignment='center',color='black',fontsize=7)


    for i in range(nrows*ncols-len(positions)-1):

        # remove the empty axes at the bottom
        ax = next(axes_iter)
        ax.remove()

    plt.subplots_adjust(wspace=-0.1, hspace=0)

    if filename:
        #plt.savefig(filename.with_suffix('.png'),dpi=300)
        plt.savefig(filename,dpi=400)
    plt.show()

def multi_cutout_rgb(positions,r,g,b,mask1=None,mask2=None,points=None,labels=None,
                 filename=None,size=6*u.arcsec,ncols=4):
    '''Plot multiple cutouts with the positoin of the clusters
    
    Parameters
    ----------

    image : NDData
        the background image that is used
    positions : SkyCoord
        the position of the cutouts
    masks : NDData
        A mask with outlines
    points : SkyCoord
        Points to mark in the image

    '''
    
    ncols = ncols
    nrows = int(np.ceil(len(positions)/ncols))

    width = two_column
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
    axes_iter = iter(axes.flatten())

    for position,label in zip(positions,labels):  

        ax = next(axes_iter)
        ax = single_cutout_rgb(ax,
                            position = position,
                            r=r,g=g,b=b,
                            mask1 = mask1,
                            mask2 = mask2,
                            label = label,
                            size  = 4*u.arcsecond)

    for i in range(nrows*ncols-len(positions)):

        # remove the empty axes at the bottom
        ax = next(axes_iter)
        ax.remove()
    
    plt.subplots_adjust(wspace=-0.1, hspace=0)

    if filename:
        plt.savefig(filename.with_suffix('.png'),dpi=300)
        #plt.savefig(filename.with_suffix('.pdf'),dpi=300)
    plt.show()



def multi_cutout_hst(positions,images,mask1=None,mask2=None,points=None,labels=None,
                     width=two_column,filename=None,size=6*u.arcsec,ncols=4):
    '''Plot multiple cutouts with the positoin of the clusters
    
    Parameters
    ----------

    image : NDData
        the background image that is used
    positions : SkyCoord
        the position of the cutouts
    masks : NDData
        A mask with outlines
    points : SkyCoord
        Points to mark in the image

    '''
    
    ncols = ncols
    nrows = int(np.ceil(len(positions)/ncols))

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
    axes_iter = iter(axes.flatten())

    for position,label in zip(positions,labels):  

        ax = next(axes_iter)
        ax = single_cutout_hst(ax,
                            position = position,
                            images=images,
                            mask1 = mask1,
                            mask2 = mask2,
                            label = label,
                            size  = 4*u.arcsecond)
        for child in ax.get_children():
            if isinstance(child, mpl.spines.Spine):
                child.set_color('#ffffff')
                child.set_linewidth(2)

    h,l = fig.axes[0].get_legend_handles_labels()
    ax = next(axes_iter)
    ax.axis('off')
    ax.legend(h[::len(h)-1],l[::(len(l)-1)],fontsize=7,loc='upper center',frameon=False)

    if True:
        length = 0.263
        ax.set(xlim=[0,1],ylim=[0,1])
        x,y = 0.2,0.2
        ax.plot([x,x+length],[y,y],color='black',marker='|',lw=1,ms=4)
        ax.text(x+0.5*length,y*1.3,'100 pc',horizontalalignment='center',color='black',fontsize=7)


    for i in range(nrows*ncols-len(positions)-1):

        # remove the empty axes at the bottom
        ax = next(axes_iter)
        ax.remove()

    plt.subplots_adjust(wspace=-0.1, hspace=0)

    if filename:
        plt.savefig(filename,dpi=300)
    plt.show()



from matplotlib.backends.backend_pdf import PdfPages
import datetime

def multi_page_cutout(positions,image,mask1=None,mask2=None,points=None,labels=None,
                 filename=None,size=6*u.arcsec,nrows=5,ncols=4,**kwargs):
    '''Plot multiple cutouts with the positoin of the clusters
    
    Parameters
    ----------

    image : NDData
        the background image that is used
    positions : SkyCoord
        the position of the cutouts
    masks : NDData
        A mask with outlines
    points : SkyCoord
        Points to mark in the image

    '''
    
    width = 8.27
    N = len(positions)
    Npage = nrows*ncols
    Npages = int(np.ceil(N/Npage))
    with PdfPages(filename.with_suffix('.pdf')) as pdf:
        
        for i in range(Npages):
            print(f'working on page {i+1} of {Npages}')

            
            sub_positions = positions[i*Npage:(i+1)*Npage]
            sub_labels = labels[i*Npage:(i+1)*Npage]
        
            fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
            axes_iter = iter(axes.flatten())

            for position,label in zip(sub_positions,sub_labels):  

                ax = next(axes_iter)
                ax = single_cutout(ax,
                                 position = position,
                                 image = image,
                                 mask1 = mask1,
                                 mask2 = mask2,
                                 label = label,
                                 points = points,
                                 size  = 4*u.arcsecond,
                                 **kwargs)

            plt.subplots_adjust(wspace=-0.1, hspace=0)
            
            # only the last page has subplots that need to be removed
            if i == int(np.ceil(N/Npage))-1:
                h,l = fig.axes[0].get_legend_handles_labels()
                ax = next(axes_iter)
                ax.axis('off')
                #ax.legend(h[::len(h)-1],l[::(len(l)-1)],fontsize=7,loc='center',frameon=False)
                ax.legend(h,l,fontsize=7,loc='center',frameon=False)

                t = ax.text(0.06,0.87,'region ID/assoc ID', transform=ax.transAxes,color='black',fontsize=8)

                for i in range(nrows*ncols-len(sub_positions)-1):
                    # remove the empty axes at the bottom
                    ax = next(axes_iter)
                    ax.axis('off')    
        
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


def multi_page_cutout_hst(positions,images,mask1=None,mask2=None,points=None,labels=None,
                          filename=None,size=6*u.arcsec,nrows=5,ncols=4,**kwargs):
    '''Plot multiple cutouts with the positoin of the clusters
    
    Parameters
    ----------

    image : NDData
        the background image that is used
    positions : SkyCoord
        the position of the cutouts
    masks : NDData
        A mask with outlines
    points : SkyCoord
        Points to mark in the image

    '''
    
    width = 8.27
    N = len(positions)
    Npage = nrows*ncols
    Npages = int(np.ceil(N/Npage))
    with PdfPages(filename.with_suffix('.pdf')) as pdf:
        
        for i in range(Npages):
            print(f'working on page {i+1} of {Npages}')

    
            sub_positions = positions[i*Npage:(i+1)*Npage]
            sub_labels = labels[i*Npage:(i+1)*Npage]
        
            fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
            axes_iter = iter(axes.flatten())

            for position,label in zip(sub_positions,sub_labels):  

                ax = next(axes_iter)
                ax = single_cutout_hst(ax,
                                 position = position,
                                 images = images,
                                 mask1 = mask1,
                                 mask2 = mask2,
                                 label = label,
                                 points = points,
                                 size  = 4*u.arcsecond,
                                 **kwargs)

            plt.subplots_adjust(wspace=-0.1, hspace=0)
            
            # only the last page has subplots that need to be removed
            if i == int(np.ceil(N/Npage))-1:
                h,l = fig.axes[0].get_legend_handles_labels()
                ax = next(axes_iter)
                ax.axis('off')
                #ax.legend(h[::len(h)-1],l[::(len(l)-1)],fontsize=7,loc='center',frameon=False)
                ax.legend(h,l,fontsize=7,loc='center',frameon=False)

                t = ax.text(0.06,0.87,'region ID/assoc ID', transform=ax.transAxes,color='black',fontsize=8)

                for i in range(nrows*ncols-len(sub_positions)-1):
                    # remove the empty axes at the bottom
                    ax = next(axes_iter)
                    ax.axis('off')    
        
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()





def plot_cluster_nebulae(name,position,size,
                         F275,Halpha,astrosat,
                         sdss_g,sdss_r,sdss_i,
                         reg_hst_sky,nebulae_mask,
                         associations_mask,
                         HII_regions,
                         filename=None):
    '''Plot a cutout of the galaxy in different filters

    1. Complete galaxy with position of the cluster/nebulae marked
    2. FUV image of the cutout
    3. Halpha image of the cutout with the outline of the HII-regions
    4. F275 image of the cutout with the outline of the HII-regions and the position of the clusters
    '''

    # create the cutouts 
    hst_cutout     = Cutout2D(F275.data,position,size=size,wcs=F275.wcs)
    HA_cutout      = Cutout2D(Halpha.data,position,size=size,wcs=Halpha.wcs)
    #OIII_cutout    = Cutout2D(OIII.data,position,size=size,wcs=OIII.wcs)
    #FUV_cutout, _  = reproject_interp(astrosat,output_projection=HA_cutout.wcs,shape_out=HA_cutout.shape)    
    FUV_cutout     = Cutout2D(astrosat.data,position,size=size,wcs=astrosat.wcs)


    fig = plt.figure(figsize=(two_column,two_column/4))
    ax1  = fig.add_subplot(141,projection=Halpha.wcs)
    ax2  = fig.add_subplot(142,projection=HA_cutout.wcs)
    ax3  = fig.add_subplot(143,projection=Halpha.wcs)
    ax4  = fig.add_subplot(144,projection=HA_cutout.wcs)
    
    gri = create_RGB(sdss_i,sdss_r,sdss_g,weights=[1,1,1],percentile=[99,99,99])

    # show image of the entire galaxy
    ax1.imshow(gri)
    reg_hst_muse  = reg_hst_sky.to_pixel(Halpha.wcs)
    reg_hst_muse.plot(ax=ax1,ec='tab:red',label='HST',lw=0.5)
    x,y = position.to_pixel(Halpha.wcs)
    ax1.scatter(x,y,marker='s',facecolors='none',s=10,lw=0.5,ec='tab:red')
    ax1.set_title(f'{name}')
    label = 'R.A.={}, Dec.={}'.format(*position.to_string(style='hmsdms',precision=2).split(' '))
    #t = ax1.text(0.05,0.9,label, transform=ax1.transAxes,color='black',fontsize=2)
    #t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))
    
    # FUV plot (rgb or alone)
    #rgb = create_RGB(HA_cutout.data,OIII_cutout.data,FUV_cutout,percentile=[99,99,99])
    #ax2.imshow(rgb)
    norm = simple_norm(FUV_cutout.data,clip=False)
    ax2.imshow(FUV_cutout.data,norm=norm,cmap=plt.cm.Blues)
    ax2.set(xlabel=size,ylabel=size)
    ax2.set_title('Astrosat (FUV)')
    
    add_scale(ax2,1*u.arcsec,label="1'")
    
    # the Halpha whitelight image
    norm = simple_norm(HA_cutout.data,clip=False,max_percent=98,stretch='asinh')
    ax3.imshow(HA_cutout.data,norm=norm,cmap=plt.cm.Reds)
    ax3.set_title(r'MUSE (H$\alpha$)')
    
    # the HST whitelight image
    norm = simple_norm(hst_cutout.data,clip=False,max_percent=99.8)
    ax4.imshow(hst_cutout.data,norm=norm,cmap=plt.cm.gray_r)
    ax4.set_title('HST (F275)')
    
    # plot the nebulae catalogue
    nebulae_mask_hst, _  = reproject_interp(nebulae_mask,output_projection=hst_cutout.wcs,shape_out=hst_cutout.shape,order='nearest-neighbor')    
    nebulae_mask_muse, _ = reproject_interp(nebulae_mask,output_projection=HA_cutout.wcs,shape_out=HA_cutout.shape,order='nearest-neighbor')    

    region_ID = np.unique(nebulae_mask_muse[~np.isnan(nebulae_mask_muse)])
    
    contours_hst_hii = []
    contours_hst_neb = []

    contours_muse_hii = []
    contours_muse_neb = []

    for i in region_ID:
        if i in HII_regions['region_ID']:
            blank_mask = np.zeros_like(nebulae_mask_hst)
            blank_mask[nebulae_mask_hst==i] = 1
            contours_hst_hii += find_contours(blank_mask, 0.5)
            
            blank_mask = np.zeros_like(nebulae_mask_muse)
            blank_mask[nebulae_mask_muse==i] = 1
            contours_muse_hii += find_contours(blank_mask, 0.5)
        else:
            blank_mask = np.zeros_like(nebulae_mask_hst)
            blank_mask[nebulae_mask_hst==i] = 1
            contours_hst_neb += find_contours(blank_mask, 0.5)
            
            blank_mask = np.zeros_like(nebulae_mask_muse)
            blank_mask[nebulae_mask_muse==i] = 1
            contours_muse_neb += find_contours(blank_mask, 0.5)

    cutout_mask, _  = reproject_interp(associations_mask,output_projection=hst_cutout.wcs,shape_out=hst_cutout.shape,order='nearest-neighbor')    
    contours_hst_asc = []
    for i in np.unique(cutout_mask[~np.isnan(cutout_mask)]):
        blank_mask = np.zeros_like(cutout_mask)
        blank_mask[cutout_mask==i] = 1
        contours_hst_asc += find_contours(blank_mask, 0.5)

    for coords in contours_muse_hii: 
        ax3.plot(coords[:,1],coords[:,0],color='black',lw=0.2)
    for coords in contours_muse_neb: 
        ax3.plot(coords[:,1],coords[:,0],ls='--',color='black',lw=0.2)
    for coords in contours_hst_hii:
        ax4.plot(coords[:,1],coords[:,0],color='tab:blue',lw=0.2)
    for coords in contours_hst_neb:
        ax4.plot(coords[:,1],coords[:,0],ls='--',color='tab:blue',lw=0.2) 
    for coords in contours_hst_asc:
        ax4.plot(coords[:,1],coords[:,0],color='tab:red',lw=0.5)

    '''
    # mark the position of the clusters within the cutout
    region = RectangleSkyRegion(position,size,size)
    clusters_in_frame = hst_all_objects[region.contains(hst_all_objects['SkyCoord'],hst_cutout.wcs)]
    for cluster in clusters_in_frame:
        x,y = cluster['SkyCoord'].to_pixel(hst_cutout.wcs)
        if 5<x<hst_cutout.data.shape[0]-5 and 5<y<hst_cutout.data.shape[1]-5:
            if cluster['CLUSTER_CLASS']<0:
                ax4.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.6,ec='tab:blue',label='cluster')
            else:
                ax4.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.4,ec='tab:green',label='unclassified object')
    '''
    
    # only add one handle per ojbect to the legend
    #handles, labels = ax1.get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #ax1.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)
 
    
    for ax in [ax1,ax2,ax3,ax4]:
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[0].set_ticks_visible(False)
        ax.coords[1].set_ticks_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)
    
    if filename:
        print(f'save image to file {filename}.pdf')
        plt.savefig(filename.with_suffix('.png'),dpi=600)
        plt.savefig(filename.with_suffix('.pdf'),dpi=600)
        
    plt.show()
    