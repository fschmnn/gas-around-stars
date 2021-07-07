import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from cluster.auxiliary import bin_stat

def corner(table,columns,limits={},colors=None,nbins=10,one_to_one=False,filename=None,figsize=(8,6),**kwargs):
    '''Create a pairwise plot for all names in columns
    
    Parameters
    ----------
    
    table : table with the data
    
    columns : name of the columns ought to be used
    
    limits : dictionary with the limits (tuple) for the columns
    filename : name of the file to save to
    '''
    
    fig, axes = plt.subplots(nrows=len(columns)-1,ncols=len(columns),figsize=figsize)

    for i,row in enumerate(columns[1:]):
        for j,col in enumerate(columns):
            ax=axes[i,j]

            if j>i:
                ax.remove()
            else:
                if j==0:
                    ax.set_ylabel(row.replace("_",""))
                else:
                    ax.set_yticklabels([])
                if i==len(columns)-2:
                    ax.set_xlabel(col.replace("_",""))
                else:
                    ax.set_xticklabels([])

                ax.scatter(table[col],table[row],**kwargs)


                xlim = limits.get(col,None)
                ylim = limits.get(row,None)
                if xlim: 
                    x,mean,std = bin_stat(table[col],table[row],xlim,nbins=nbins)
                    #ax.errorbar(x,mean,yerr=std,fmt='-',color='black')
                    ax.set_xlim(xlim)

                if ylim: ax.set_ylim(ylim)


                if one_to_one:
                    if not xlim:
                        xlim = ax.get_xlim()
                    if not ylim:
                        ylim = ax.get_ylim()
                    lim = np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]])
                    ax.plot(lim,lim,color='black')
                    ax.set(xlim=lim,ylim=lim)

                not_nan = ~np.isnan(table[col]) & ~np.isnan(table[row])
                r,p = spearmanr(table[col][not_nan],table[row][not_nan])
    
                t = ax.text(0.06,0.89,r'$\rho$'+f'={r:.2f}',transform=ax.transAxes,fontsize=7)
                t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))


    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    if filename:
        plt.savefig(filename,dpi=600)

    plt.show()