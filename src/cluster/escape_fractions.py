'''Functions to compute the escape fraction of HII regions


'''

import numpy as np
import matplotlib.pyplot as plt 
from astrotools.constants import tab10, single_column, two_column, thesis_width


def plot_ionising_photon_flux(age,age_err,model_age,model_flux,sample_size=10000,
                              xlim=[0.5,50],ylim=[5e48,5e53]):
    '''plot the ionising  photon flux at a certain age
    
    '''
    


    
    age_arr  = np.random.normal(loc=age,scale=age_err,size=(sample_size))
    idx = np.argmin(np.abs(age_arr[:,np.newaxis]-model_age),axis=1)
    flux_arr = model_flux[idx]
    flux_err = np.std(model_flux[idx])
    flux = model_flux[np.argmin(np.abs(age-model_age))]

    fig=plt.figure(figsize=(thesis_width,thesis_width/1.618))

    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(3, 7),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.03, hspace=0.03)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.plot(model_age,model_flux)
    ax.plot([age,age,100],[1e60,flux,flux],ls='--',color='gray')
    
    ax_histx.hist(age_arr,bins=np.logspace(*np.log10(xlim),50),
                  color='#af1d1d',histtype='step',density=False)
    ax_histy.axhline(flux,ls='--',color='gray')

    ax_histy.hist(flux_arr,bins=np.logspace(*np.log10(ylim),50),
                  orientation='horizontal',color='#af1d1d',histtype='step',density=False)
    ax_histx.axvline(age,ls='--',color='gray')    

    ax.text(0.04,0.4,f'age:  $({age}\pm{age_err})$ Myr', transform=ax.transAxes,color='black',fontsize=10)
    exp = np.floor(np.log10(flux))
    msg = f'flux: $({flux/10**exp:.1f}\pm{flux_err/10**exp:.1f})\cdot 10^{{{exp:.0f}}}$ s$^{{-1}}$'
    ax.text(0.04,0.3,msg, transform=ax.transAxes,color='black',fontsize=10)
 
    
    ax.set(xlim=xlim,xscale='log',xlabel=r'age / Myr',
           ylim=ylim,yscale='log',ylabel='$Q$ / s$^{-1}$')

    plt.show()


def ionising_photon_flux(age,age_err,model_age,model_flux,sample_size=1000):
    '''compute the ionising photon flux at a certain age
    
    
    Parameters
    ----------
    
    age : float or numpy.array
    
    age_err : flaot or numpy.array
    
    model : starburst99 model
    
    '''
    
    age     = np.atleast_1d(age)
    age_err = np.atleast_1d(age_err)

    # compute the photon flux according to the model
    flux = model_flux[np.argmin(np.abs(age[:,np.newaxis]-model_age),axis=1)]
    
    # sample the catalogue
    sample_ext = np.random.normal(loc=age,scale=age_err,size=(sample_size,len(age)))
    # broadcasting to np.newpaxis is much faster than a loop
    idx = np.argmin(np.abs(sample_ext[:,:,np.newaxis]-model_age),axis=2)
    flux_err = np.std(model_flux[idx],axis=0)

    return flux, flux_err


def compare_ionising_photons(Qpredicted,Qobserved,ax=None,plot_lines=True,**kwargs):
    '''plot Qpredicted against Qobserved
    
    Parameters
    ----------
    Qpredicted : array

    Qobserved : array

    ax : matplotlib.axes._subplots.AxesSubplot
        use an existing axes

    plot_lines : bool
        plot lines for different escape fractions
    '''
    
    if not ax:
        fig,ax =plt.subplots(figsize=(single_column,single_column))

    if plot_lines:
        Qpredicted_line = np.logspace(48,52)
        cmap = plt.cm.get_cmap('cool',6)
        lines = ["-","--","-.",":"]
        for i,f in enumerate([0.0,0.5,0.9,0.99]):
            Qobserved_line = Qpredicted_line*(1-f)
            ax.plot(np.log10(Qpredicted_line),np.log10(Qobserved_line),ls=lines[i],c='k',label=f'$f_\mathrm{{esc}}={f}$',zorder=1)
        ax.fill_between(np.log10(Qpredicted_line),4*np.log10(Qpredicted_line),np.log10(Qpredicted_line),color='0.7',alpha=0.1)
        
    sc=ax.scatter(np.log10(Qpredicted),np.log10(Qobserved),rasterized=True,**kwargs)
    #fig.colorbar(sc,label='density / cm$^{-3}$')

    ax.set(xlabel=r'$\log_{10} (Q (\mathrm{H}^0)\,/\,\mathrm{s}^{-1})$ predicted',
           ylabel=r'$\log_{10} (Q_{\mathrm{H}\,\alpha}\,/\,\mathrm{s}^{-1})$ observed',
           xlim=[49,52],ylim=[49,52])

    ax.set_xticks([49,50,51,52])
    ax.set_yticks([49,50,51,52])
    
    
    return ax


def escape_fraction(Qpredicted,Qpredicted_err,Qobserved,Qobserved_err,stats=False):
    '''compute escape fraction
    
    
    Parameters 
    ----------
    
    Qpredicted : array
        the predicted ionising photon flux
    Qpredicted_err : array
        error of the predicted ionising photon flux
    Qobserved : array
        the observed ionising photon flux
    Qobserved_err : array
        error of the observed ionising photon flux 
    stats : bool
        print some stats
    '''
    
    fesc = (Qpredicted-Qobserved)/Qpredicted
    fesc_err = Qobserved/Qpredicted*np.sqrt((Qobserved_err/Qobserved)**2+(Qpredicted_err/Qpredicted)**2)
    
    if stats:
        print(f'{len(fesc)} objects in initial catalogue')
        print(f'{np.sum(fesc<0)} HII regions ({np.sum(fesc<0)/len(fesc)*100:.1f} %) have negative escape fractions')
        print(f'{np.sum(Qpredicted+Qpredicted_err<Qobserved)} are inconsistent within uncertainties')
        # we only accept objects that are physical reasonable
        clean = fesc[Qpredicted+Qpredicted_err>=Qobserved]
        clean[clean<0] = 0
        l,m,h=np.percentile(clean,[16,50,84])
        label = f'fesc={m:.2f}+{h-m:.2f}-{m-l:.2f}$'
        print(label)
        
    return fesc, fesc_err


def plot_escape_fraction(Qpredicted,Qpredicted_err,Qobserved,Qobserved_err,ax=None):
    '''plot an histogram of the escape fraction
    
    
    Parameters 
    ----------
    
    Qpredicted : array
        the predicted ionising photon flux
    Qpredicted_err : array
        error of the predicted ionising photon flux
    Qobserved : array
        the observed ionising photon flux
    Qobserved_err : array
        error of the observed ionising photon flux 
    
    '''

    if not ax:
        fig,ax=plt.subplots(figsize=(single_column,single_column/1.618))
    
    fesc = (Qpredicted-Qobserved)/Qpredicted
    fesc_err = Qobserved/Qpredicted*np.sqrt((Qobserved_err/Qobserved)**2+(Qpredicted_err/Qpredicted)**2)
    
    # we only accept objects that are physical reasonable
    clean = fesc[Qpredicted+Qpredicted_err>=Qobserved]
    clean[clean<0] = 0
    l,m,h=np.percentile(clean,[16,50,84])
    label = f'$f_\mathrm{{esc}}={m:.2f}^{{+{h-m:.2f}}}_{{-{m-l:.2f}}}$'

    ax.hist(100*clean,bins=np.linspace(0,100,20),histtype='step')
    ax.text(0.15,0.85,label, transform=ax.transAxes,color='black',fontsize=8)
    ax.set(xlabel=r'$f_\mathrm{esc}$ / per\,cent',xlim=[-5,105])
        
    return ax




    
