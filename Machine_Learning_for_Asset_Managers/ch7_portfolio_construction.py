# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import matplotlib.pylab as plt
import matplotlib.pyplot as mp1
import seaborn as sns

from Machine_Learning_for_Asset_Managers import ch2_monte_carlo_experiment as mc
from Machine_Learning_for_Asset_Managers import ch2_marcenko_pastur_pdf as mp
from Machine_Learning_for_Asset_Managers import ch4_optimal_clustering as oc


def minVarPort(cov):
    return mc.optPort(cov, mu = None)

# code snippet 7.6 - function implementing the NCO algorithm
# Long only portfolio uses allocate_cvo()
# Method assumes input - detoned covariance matrix
def optPort_nco(cov, mu=None, maxNumClusters=None):
    cov = pd.DataFrame(cov)
    if mu is not None:
        mu = pd.Series(mu[:,0])
    
    corr1 = mp.cov2corr(cov)
    
    # Optimal partition of clusters (step 1)
    corr1, clstrs, _ = oc.clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    #wIntra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    w_intra_clusters = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_cluster = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None:
            mu_cluster = None
        else: 
            mu_cluster = mu.loc[clstrs[i]].values.reshape(-1,1)
        
        #Long/Short
        #w_intra_clusters.loc[clstrs[i],i] = mc.optPort(cov_cluster, mu_cluster).flatten()
        
        # Long only: Estimating the Convex Optimization Solution in a cluster (step 2)
        w_intra_clusters.loc[clstrs[i], i] = allocate_cvo(cov_cluster, mu_cluster).flatten()        
    
    cov_inter_cluster = w_intra_clusters.T.dot(np.dot(cov, w_intra_clusters)) #reduce covariance matrix
    mu_inter_cluster = (None if mu is None else w_intra_clusters.T.dot(mu))
    
    #Long/Short
    #w_inter_clusters = pd.Series(mc.optPort(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)
    # Long only: Optimal allocations across the reduced covariance matrix (step 3)
    w_inter_clusters = pd.Series(allocate_cvo(cov_inter_cluster, mu_inter_cluster).flatten(), index=cov_inter_cluster.index)    
    
    # Final allocations - dot-product of the intra-cluster and inter-cluster allocations (step 4)
    nco = w_intra_clusters.mul(w_inter_clusters, axis=1).sum(axis=1).values.reshape(-1,1)
    return nco
    
def allocate_cvo(cov, mu_vec=None):
    """
    Estimates the Convex Optimization Solution (CVO).
    Uses the covariance matrix and the mu - optimal solution.
    If mu is the vector of expected values from variables, the result will be
    a vector of weights with maximum Sharpe ratio.
    If mu is a vector of ones, the result will be a vector of weights with
    minimum variance.
    :param cov: (np.array) Covariance matrix of the variables.
    :param mu_vec: (np.array) Expected value of draws from the variables for maximum Sharpe ratio.
                          None if outputting the minimum variance portfolio.
    :return: (np.array) Weights for optimal allocation.
    """
    
    # Calculating the inverse covariance matrix
    inv_cov = np.linalg.inv(cov)
    
    # Generating a vector of size of the inverted covariance matrix
    ones = np.ones(shape=(inv_cov.shape[0], 1))
    
    if mu_vec is None:  # To output the minimum variance portfolio
        mu_vec = ones
    
    # Calculating the analytical solution using CVO - weights
    w_cvo = np.dot(inv_cov, mu_vec)
    w_cvo /= np.dot(mu_vec.T, w_cvo)
    
    return w_cvo    
   
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
       
    # code snippet 7.8 - drawing an empirical vector of means and covariance matrix
    nObs, nSims, shrink, minVarPortf = 1000, 1000, False, True
    np.random.seed(0)
    w1 = pd.DataFrame(0, index=range(0, nSims), columns=range(0, nBlocks*bSize))	
    w1_d = pd.DataFrame(0, index=range(0, nSims), columns=range(0, nBlocks*bSize))
    for i in range(0, nSims):
        mu1, cov1 = mc.simCovMu(mu0, cov0, nObs, shrink=shrink)
        if minVarPortf:
            mu1 = None
        w1.loc[i] = mc.optPort(cov1, mu1).flatten() #markowitc
        w1_d.loc[i] = optPort_nco(cov1, mu1, int(cov1.shape[0]/2)).flatten() #nco
        
    # code snippet 7.9 - Estimation of allocation errors
    w0 = mc.optPort(cov0, None if minVarPortf else mu0)
    w0 = np.repeat(w0.T, w1.shape[0], axis=0) #true allocation
    rmsd = np.mean((w1-w0).values.flatten()**2)**.5 #RMSE
    rmsd_d = np.mean((w1_d-w0).values.flatten()**2)**.5 #RMSE
    '''
    >>> rmsd
    0.020737753489610305 #markowitc
    >>> rmsd_d
    0.015918559234396952 #nco
    '''
    
    
    