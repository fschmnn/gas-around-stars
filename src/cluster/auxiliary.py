import numpy as np
import matplotlib.pyplot as plt 


def find_closest_point_on_curve(point,curve,conversion_factor=1,plot=False):
    '''The the closest point on a 2D curve
    
    assumes that both axis have the same unit (or use conversion_factor)
    
    Parameters
    ----------
    
    point : (float,float) / (array/array)
        
    curve : (array,array)
        
    conversion_factor : float
        if x and y have different units: y = x*conversion_factor
    '''
    
    x_point,y_point=point
    x_curve,y_curve=curve
    
    x_point = np.atleast_1d(x_point)
    y_point = np.atleast_1d(y_point)
    
    distance = np.sqrt((x_curve[:,None]-x_point)**2 + conversion_factor*(y_curve[:,None]-y_point)**2)
    idx = np.argmin(distance,axis=0)

    if plot:
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot(x_curve,y_curve,color='black')
        #ax.plot(x_curve,distance,color='grey')
        for xp,yp,i in zip(x_point,y_point,idx):
            ax.plot([xp,x_curve[i]],[yp,y_curve[i]])
            ax.scatter(xp,yp)
        #ax.set_aspect('equal')
        plt.show()
        
    return x_curve[idx],y_curve[idx]

if __name__ == '__main__':

    # showcase the find_closest_point_on_curve function
    find_closest_point_on_curve(([3,7,12],[8,14,8]),
                            (np.linspace(0,15),2+15/(1+np.exp(np.linspace(0,15)-5))),
                             conversion_factor=1,plot=True)