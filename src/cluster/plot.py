import numpy as np
import matplotlib.pyplot as plt

from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u


from reproject import reproject_interp
from regions import PixCoord, RectangleSkyRegion
from skimage.measure import find_contours

from pnlf.plot import add_scale, create_RGB

single_column = 3.321 # in inch
two_column    = 6.974 # in inch

def plot_cutouts(positions,image,mask,mask_ID,points,labels,
                 filename=None,size=6*u.arcsec,ncols=4):
    '''
    
    
    '''
    
    ncols = ncols
    nrows = int(np.ceil(len(positions)/ncols))

    width = two_column
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
    axes_iter = iter(axes.flatten())

    for position,label in zip(positions,labels):  

        ax = next(axes_iter)

        cutout_image = Cutout2D(image.data,position,size=size,wcs=image.wcs)
        norm = simple_norm(cutout_image.data,clip=False,max_percent=99)
        ax.imshow(cutout_image.data,norm=norm,cmap=plt.cm.gray_r)

        # plot the nebulae catalogue
        cutout_mask, _  = reproject_interp(mask,output_projection=cutout_image.wcs,shape_out=cutout_image.shape,order='nearest-neighbor')    
        region_ID = np.unique(cutout_mask[~np.isnan(cutout_mask)])

        contours_true = []
        contours_false = []

        for i in region_ID:
            if i in mask_ID:
                blank_mask = np.zeros_like(cutout_mask)
                blank_mask[cutout_mask==i] = 1
                contours_true += find_contours(blank_mask, 0.5)
            else:
                blank_mask = np.zeros_like(cutout_mask)
                blank_mask[cutout_mask==i] = 1
                contours_false += find_contours(blank_mask, 0.5)

        for coords in contours_true:
            ax.plot(coords[:,1],coords[:,0],color='tab:red',lw=0.5)
        for coords in contours_false:
            ax.plot(coords[:,1],coords[:,0],ls='--',color='tab:red',lw=0.5)

        # mark the position of the clusters within the cutout
        region = RectangleSkyRegion(position,0.9*size,0.9*size)
        in_frame = points[region.contains(points['SkyCoord'],cutout_image.wcs)]
        for row in in_frame:
            x,y = row['SkyCoord'].to_pixel(cutout_image.wcs)
            if 5<x<cutout_image.data.shape[0]-5 and 5<y<cutout_image.data.shape[1]-5:
                if row['CLUSTER_CLASS']<4:
                    ax.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.4,color='tab:blue',label='cluster')
                else:
                    ax.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.4,color='tab:green',label='unclassified object')

        plt.subplots_adjust(wspace=-0.1, hspace=0)

        t = ax.text(0.05,0.89,f'{label:.0f}', transform=ax.transAxes,color='black',fontsize=8)
        t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))

        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(nrows*ncols-len(positions)):

        # remove the empty axes at the bottom
        ax = next(axes_iter)
        ax.remove()

    if filename:
        plt.savefig(filename.with_suffix('.png'),dpi=600)
        plt.savefig(filename.with_suffix('.pdf'),dpi=600)
    plt.show()


def plot_cluster_nebulae(name,position,size,
                         F275,Halpha,astrosat,
                         sdss_g,sdss_r,sdss_i,
                         reg_hst_sky,nebulae_mask,
                         HII_regions,hst_all_objects,
                         filename=None):
    '''Plot a cutout of the galaxy in different filters

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

    for coords in contours_muse_hii: 
        ax3.plot(coords[:,1],coords[:,0],color='black',lw=0.2)
    for coords in contours_muse_neb: 
        ax3.plot(coords[:,1],coords[:,0],ls='--',color='black',lw=0.2)
    for coords in contours_hst_hii:
        ax4.plot(coords[:,1],coords[:,0],color='tab:red',lw=0.2)
    for coords in contours_hst_neb:
        ax4.plot(coords[:,1],coords[:,0],ls='--',color='tab:red',lw=0.2) 

    # mark the position of the clusters within the cutout
    region = RectangleSkyRegion(position,size,size)
    clusters_in_frame = hst_all_objects[region.contains(hst_all_objects['SkyCoord'],hst_cutout.wcs)]
    for cluster in clusters_in_frame:
        x,y = cluster['SkyCoord'].to_pixel(hst_cutout.wcs)
        if 5<x<hst_cutout.data.shape[0]-5 and 5<y<hst_cutout.data.shape[1]-5:
            if cluster['CLUSTER_CLASS']<4:
                ax4.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.6,ec='tab:blue',label='cluster')
            else:
                ax4.scatter(x,y,marker='o',facecolors='none',s=20,lw=0.4,ec='tab:green',label='unclassified object')

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
    