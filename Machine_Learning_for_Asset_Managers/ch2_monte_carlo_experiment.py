# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf

from Machine_Learning_for_Asset_Managers import ch2_marcenko_pastur_pdf as mp

#import cvxpy as cp

# Code snippet 2.7
#Generate a block-diagnoal covariance matrix and a vector of means
def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones( (bSize, bSize))*bCorr
    block[range(bSize), range(bSize)] = 1 #diagonal is 1
    corr = block_diag(*([block]*nBlocks))
    return corr
    
def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1,1)
    return mu0, cov0
    
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov
    
# Code snippet 2.8
# generating the empirical covariance matrix
def simCovMu(mu0, cov0, nObs, shrink=False):
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size = nObs)
    #print(x.shape)
    mu1 = x.mean(axis = 0).reshape(-1,1) #calc mean of columns of rand matrix
    #print(mu1.shape)
    if shrink: cov1 = LedoitWolf().fit(x).covariance_
    else: cov1 = np.cov(x, rowvar=0)
    return mu1, cov1

# code snippet 2.9 
# Denoising of the empirical covariance matrix
# by constant residual eigenvalue method
def deNoiseCov(cov0, q, bWidth):
    corr0 = mp.cov2corr(cov0)
    eVal0, eVec0 = mp.getPCA(corr0)
    eMax0, var0 = mp.findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = mp.denoisedCorr(eVal0, eVec0, nFacts0) #denoising by constant residual eigenvalue method
    cov1 = corr2cov(corr1, np.diag(cov0)**.5)
    return cov1
    
# code snippet 2.10
# Derive minimum-variance-portfolio
# Returns a column vector of percentage allocations
# should be subject to lagrangian constraints:
# 1. lambda_1*(sum(expectation(x_i)*x_i) - d = 0
# 2. lambda_2*(sum(x_i - 1))=0
# where d is expected rate of return
# w*=C^−1*μ/I.T*C^−1*μ - is minimum-variance-portfolio
 #short sales are allowed
def optPort(cov, mu = None):
    inv = np.linalg.inv(cov) #The precision matrix: contains information about the partial correlation between variables,
    #  the covariance between pairs i and j, conditioned on all other variables (https://www.mn.uio.no/math/english/research/projects/focustat/publications_2/shatthik_barua_master2017.pdf)
    ones = np.ones(shape = (inv.shape[0], 1)) # column vector 1's
    if mu is None: 
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w) # def: w = w / sum(w) ~ w is column vector
    
    return w
    
#optPort with long only curtesy of Brady Preston
#requires: import cvxpy as cp
'''def optPort(cov,mu=None):
    n = cov.shape[0]
    if mu is None:mu = np.abs(np.random.randn(n, 1))
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov)
    ret =  mu.T @ w
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(risk),constraints)
    prob.solve(verbose=True)
    return np.array(w.value.flat).round(4)'''

#According to the question 'Tangent portfolio weights without short sales?' 
#there is no analytical solution to the GMV problem with no short-sales constraints
#So - set the negative weights in WGV to 0, and make w sum up to 1
def optPortLongOnly(cov, mu = None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape = (inv.shape[0], 1)) # column vector 1's
    if mu is None: 
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w) # def: w = w / sum(w) ~ w is column vector
    w = w.flatten()
    threshold = w < 0
    wpluss = w.copy()
    wpluss[threshold] = 0
    wpluss = wpluss/np.sum(wpluss)
    
    return wpluss
    
if __name__ == '__main__':
    nBlocks, bSize, bCorr = 2, 2, .5
    np.random.seed(0)
    mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)

    # code snippet 2.10
    nObs, nTrials, bWidth, shrink, minVarPortf = 5, 5, .01, False, True
    w1 = pd.DataFrame(columns = range(cov0.shape[0]), index = range(nTrials), dtype=float)

    w1_d = w1.copy(deep=True)
    np.random.seed(0)
    for i in range(nTrials):
        mu1, cov1 = simCovMu(mu0, cov0, nObs, shrink = shrink)
        if minVarPortf: mu1 = None
        cov1_d = deNoiseCov(cov1, nObs*1./cov1.shape[1], bWidth)
        w1.loc[i] = optPort(cov1, mu1).flatten() # add column vector w as row in w1
        w1_d.loc[i] = optPort(cov1_d, mu1).flatten() # np.sum(w1_d, axis=1) is vector of 1's. sum(np.sum(w1_d, axis=0)= nTrials
        # so minimum-variance-portfolio is 1./nTrials*(np.sum(w1_d, axis=0)) - but distribution not stationary
    
    min_var_port = 1./nTrials*(np.sum(w1_d, axis=0)) 
    #code snippet 2.11
    w0 = optPort(cov0, None if minVarPortf else mu0) # w0 true percentage asset allocation
    w0 = np.repeat(w0.T, w1.shape[0], axis=0) 
    rmsd = np.mean((w1-w0).values.flatten()**2)**.5     #RMSE not denoised
    rmsd_d = np.mean((w1_d-w0).values.flatten()**2)**.5 #RMSE denoised
    print("RMSE not denoised:"+str( rmsd))
    print("RMSE denoised:"+str( rmsd_d))
    