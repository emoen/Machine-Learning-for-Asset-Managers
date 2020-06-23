import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity

def mpPDF(var, q, pts):
    #Marcenko-Pastur pdf
    #q=T/N
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf, index=eVal)
    return pdf
    
#Test Marcenko-Pastur Thm
def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] #arguments for sorting eval desc
    eVal,eVec = eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec
    
def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    #Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) ==1: obs=obs.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None: x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1: x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb), index=x.flatten())
    return pdf
    
if __name__ == '__main__':
    x = np.random.normal(size = (10000, 1000))
    cor = np.corrcoef(x, rowvar=0) # cor.shape = (1000,1000). If rowvar=1 - row represents a var, with observations in the columns.
    eVal0 , eVec0 = getPCA( cor ) 
    pdf0 = mpPDF(1., q=x.shape[0]/float(x.shape[1]), pts=1000)
    pdf1 = fitKDE(np.diag(eVal0), bWidth=.01) #empirical pdf
    
    #import doctest
    #doctest.testmod()
