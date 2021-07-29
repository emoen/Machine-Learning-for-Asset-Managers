# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

# Marginal-, joint-distribution, conditional entropies, and mutual information
# https://pypi.org/project/pyitlib/

#codesnippet 3.3
#Variation of information on discretized continuous random variables
def numBins(nObs, corr=None):
    #optimal number of bins for discretization
    if corr is None: #univariate case
        z = (8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1/3.)
        b = round(z/6.+2./(3*z)+1./3)
    else: #bivariate case
        b = round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    
    return int(b)

#codesnippet 3.2
def varInfo(x,y, bins, norm=False):
    #variation of information
    bXY = numBins(x.shape[0], corr= np.corrcoef(x,y)[0,1])
    bins = bXY
    cXY = np.histogram2d(x, y, bins)[0]
    hX = ss.entropy(np.histogram(x, bins)[0]) #marginal 
    hY = ss.entropy(np.histogram(y, bins)[0]) #marginal
    iXY = mutual_info_score(None, None, contingency=cXY)
    vXY = hX+hY-2*iXY #variation of information
    if norm:
        hXY = hX + hY - iXY #joint
        vXY = vXY/hXY #normalized varaiation of information - Kraskov (2008)
        
    return vXY

#codesnippet 3.4 Correlation and normalized mutual information of two independent gaussian random variables
def mutualInfor(x,y, norm=False):
    #mutual information
    bXY = numBins(x.shape[0], corr = np.corrcoef(x,y)[0,1])
    cXY = np.histogram2d(x,y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    if norm:
        hX = ss.entropy(np.histogram(x, bins)[0]) #marginal 
        hY = ss.entropy(np.histogram(y, bins)[0]) #marginal
        iXY /= min(hX, hY) #normalized mutual information

    return iXY
    
if __name__ == '__main__':
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    bins=10 # descretize sample space

    #codesnippet 3.1
    cXY = np.histogram2d(x, y, bins)[0]
    hX = ss.entropy(np.histogram(x, bins)[0]) #marginal 
    hY = ss.entropy(np.histogram(y, bins)[0]) #marginal
    iXY = mutual_info_score(None, None, contingency=cXY)
    iXYn = iXY/min(hX, hY) #normalized mutual information
    hXY = hX+hY - iXY #joint
    hX_Y = hXY-hY #conditional
    hY_X = hXY-hX #contitional
    
    #codesnippet 3.4
    size, seed = 5000, 0 
    np.radom.seed(seed)
    x = np.random.normal(size=size)
    e = np.random.normal(size=size)
    y = 0*x+e
    nmi = mutualInfo(x,y,True)
    corr = np.corrcoef(x,y)[0,1]

