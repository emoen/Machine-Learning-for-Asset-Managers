import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pylab as plt
from scipy.optimize import minimize

#snippet 2.1
#Marcenko-Pastur pdf
#q=T/N 
def mpPDF(var, q, pts):
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2 # calc lambda_minus, lambda_plus
    eVal = np.linspace(eMin, eMax, pts) #Return evenly spaced numbers over a specified interval. eVal='lambda'
    #Note: 1.0/2*2 = 1.0 not 0.25=1.0/(2*2)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 #np.allclose(np.flip((eMax-eVal)), (eVal-eMin))==True
    pdf=pd.Series(pdf, index=eVal)
    return pdf

#snippet 2.2
#Test Marcenko-Pastur Thm
def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec=np.linalg.eig(matrix) #complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    indices=eVal.argsort()[::-1] #arguments for sorting eval desc
    eVal,eVec = eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal) # identity matrix with eigenvalues as diagonal
    return eVal,eVec
    
def fitKDE(obs, bWidth=.15, kernel='gaussian', x=None):
    #Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    print(len(obs.shape) == 1)
    if len(obs.shape) == 1: obs = obs.reshape(-1,1)
    kde = KernelDensity(kernel = kernel, bandwidth = bWidth).fit(obs)
    print(x is None)
    if x is None: x = np.unique(obs).reshape(-1,1)
    print(len(x.shape))
    if len(x.shape) == 1: x = x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

#snippet 2.3
def getRndCov(nCols, nFacts): #nFacts - contains signal out of nCols
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T) #random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols)) #full rank cov
    return cov

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    corr[corr<-1], corr[corr>1] = -1,1 #for numerical errors
    return corr
    
#snippet 2.4 - fitting the marcenko-pastur pdf - find variance
#Fit error
def errPDFs(var, eVal, q, bWidth, pts=1000):
    pdf0 = mpPDF(var, q, pts) #theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) #empirical pdf
    sse = np.sum((pdf1-pdf0)**2)
    print("sse:"+str(sse))
    return sse
    
#find max random eVal by fitting Marcenko's dist
def findMaxEval(eVal, w, bWidth):
    bnds = ((float(1e5/10000000000), float(0.99999*-1)),)
    print(bnds)
    out = minimize(lambda*x: errPDFs(*x), .5, args=(eVal, q, bWidth), bounds=bnds)
    print(out['success'])
    print(out['x'][0])
    if out['success']: var = out['x'][0]
    else: var=1
    eMax = var*(1+(1./q)**.5)**2
    return eMax, var
    
if __name__ == '__main__':
    # code snippet 2.2 - marcenko-pastur pdf explains eigenvalues of random matrix x
    N = 1000
    T = 10000
    x = np.random.normal(0, 1, size = (T, N))
    cor = np.corrcoef(x, rowvar=0) # cor.shape = (1000,1000). If rowvar=1 - row represents a var, with observations in the columns.
    eVal0 , eVec0 = getPCA( cor ) 
    pdf0 = mpPDF(1., q=x.shape[0]/float(x.shape[1]), pts=N)
    pdf1 = fitKDE(np.diag(eVal0), bWidth=.005) #empirical pdf
        
    # code snippet 2.3 - random matrix with signal
    alpha, nCols, nFact, q = .995, 1000, 100, 10
    cov = np.cov(np.random.normal(size=(nCols*q, nCols)), rowvar=0) #size = (1000*10,1000)
    cov = alpha*cov+(1-alpha)*getRndCov(nCols, nFact) # noise + signal
    corr0 = cov2corr(cov)
    eVal01, eVec01 = getPCA(corr0)
    pdf2 = fitKDE(np.diag(eVal01), bWidth=.15) #empirical pdf

    # Figure 2.1 Plot empirical:KDE and Marcenko-Pastur, and histogram
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    bins = 50
    ax.hist(np.diag(eVal0), normed = True, bins=50) # Histogram the eigenvalues

    plt.plot(pdf0.keys(), pdf0, color='r', label="Marcenko-Pastur pdf")
    plt.plot(pdf1.keys(), pdf1, color='g', label="Empirical:KDE")
    #plt.plot(x_range, pdf2, color='b', label="Eigenvalues of random-matrix with signal")
    plt.legend(loc="upper right")
    plt.show()
    
    # code snippet 2.4 
eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=.01)
nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
