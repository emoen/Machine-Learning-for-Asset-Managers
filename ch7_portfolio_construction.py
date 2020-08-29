# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import matplotlib.pyplot as mp1
import seaborn as sns

import ch2_monte_carlo_experiment as mc
import ch2_marcenko_pastur_pdf as mp
import ch4_optimal_clustering as oc


def minVarPort(cov):
    return mc.optPort(cov, mu = None)

# code snippet 7.6 - function implementing the NCO algorithm
def optPort_nco(cov, mu=None, maxNumClusters=None):
    cov = pd.DataFrame(cov)
    if mu is not None:
        mu = pd.Series(mu[:,0])
    
    corr1 = mp.cov2corr(cov)
    corr1, clstrs, _ = oc.clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    wIntra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_ = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None:
            mu_ = None
        else: 
            mu_ = mu.loc[clstrs[i]].values.reshape(-1,1)
        wIntra.loc[clstrs[i],i] = mc.optPort(cov_, mu_).flatten()
    
    cov_ = wIntra.T.dot(np.dot(cov, wIntra)) #reduce covariance matrix
    mu_ = (None if mu is None else wIntra.T.dot(mu))
    wInter = pd.Series(mc.optPort(cov_, mu_).flatten(), index=cov_.index)
    nco = wIntra.mul(wInter, axis=1).sum(axis=1).values.reshape(-1,1)
    return nco
   
if __name__ == '__main__': 
    # code snippet 7.1 - Composition of block-diagonal correlation matric
    corr0 = mc.formBlockMatrix(2, 2, .5)
    eVal, eVec = np.linalg.eigh(corr0)
    matrix_condition_number = max(eVal)/min(eVal)
    print(matrix_condition_number) 

    fig, ax = plt.subplots(figsize=(13,10))  
    sns.heatmap(corr0, cmap='viridis')
    plt.show()

    # code snippet 7.2 - block-diagonal correlation matrix with a dominant block
    corr0 = block_diag(mc.formBlockMatrix(1,2, .5))
    corr1 = mc.formBlockMatrix(1,2, .0)
    corr0 = block_diag(corr0, corr1)
    eVal, eVec = np.linalg.eigh(corr0)
    matrix_condition_number = max(eVal)/min(eVal)
    print(matrix_condition_number) 
    
    fig, ax = plt.subplots(figsize=(13,10))  
    sns.heatmap(corr1, cmap='viridis')
    plt.show()

    # code snippet 7.3 - NCO method. Step 1. Correlation matrix clustering
    nBlocks, bSize, bCorr = 2, 2, .5
    q = 10.0
    np.random.seed(0)
    mu0, cov0 = mc.formTrueMatrix(nBlocks, bSize, bCorr)
    cols = cov0.columns
    cov1 = mc.deNoiseCov(cov0, q, bWidth=.01) #denoise cov
    cov1 = pd.DataFrame(cov1, index=cols, columns=cols)
    corr1 = mp.cov2corr(cov1)
    corr1, clstrs, silh = oc.clusterKMeansBase(pd.DataFrame(corr0))
    
    # code snippet 7.4 - intracluster optimal allocations
    # step 2. compute intracluster allocations using the denoised cov matrix
    wIntra = pd.DataFrame(0, index=cov0.index, columns=clstrs.keys())
    for i in clstrs:
        wIntra.loc[clstrs[i], i] = minVarPort(cov1.loc[clstrs[i], clstrs[i]]).flatten()
        
    cov2 = wIntra.T.dot(np.dot(cov1, wIntra)) #reduced covariance matrix
    
    # code snippet 7.5 - intercluster optimal allocations
    # step 3. compute optimal intercluster allocations, usint the reduced covariance matrix
    # which is close to a diagonal matrix, so optimization problem is close to ideal case \ro =0
    wInter = pd.Series(minVarPort(cov2).flatten(), index=cov2.index)
    wAll0 = wIntra.mul(wInter, axis=1).sum(axis=1).sort_index()

    # step 4. Final allocations - dot-product of the intra-cluster and inter-cluster allocations 
    #w_nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1, 1)
    nco = wIntra.mul(wInter, axis=1).sum(axis=1).values.reshape(-1,1)
    
    # code snippet 7.7 - data-generating process
    nBlocks, bSize, bCorr = 10, 50, .5
    np.random.seed(0)
    mu0, cov0 = mc.formTrueMatrix(nBlocks, bSize, bCorr)

    # code snippet 7.7 - Drawing an empirical vector of means and covariance matrix
    nObs, nSims, shrink, minVarPortf = 1000, 1000, False, True
    np.random.seed(0)
    w1 = pd.DataFrame(0, index=range(0, nSims), columns=range(0, len(cov1[1])))
    w1_d = pd.DataFrame(0, index=range(0, nSims), columns=range(0, len(cov1[1])))
    for i in range(0, nSims):
        mu1, cov1 = mc.simCovMu(mu0, cov0, nObs, shrink=shrink)
        if minVarPortf:
            mu1 = None
        w1.loc[i] = mc.optPort(cov1, mu1).flatten()
        w1_d.loc[i] = optPort_nco(cov1, mu1, int(cov1.shape[0]/2)).flatten()
        
    # code snippet 7.8 - drawing an empirical vector of means and covariance matrix
    nObs, nSims, shrink, minVarPortf = 1000, 1000, False, True
    np.random.seed(0)
    for i in range(0, nSims):
        mu1, cov1 = mc.simCovMu(mu0, cov0, nObs, shrink=shrink)
        if minVarPortf:
            mu1 = None
        w1.loc[i] = mc.optPort(cov1, mu1).flatten()
        w1_d.loc[i] = optPort_nco(cov1, mu1, int(cov1.shape[0]/2)).flatten()
    
    