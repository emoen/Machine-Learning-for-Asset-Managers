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
    wIntra = pd.DataFrame()
    for i in clstrs:
        wIntra.loc[clstrs[i], i] = minVarPort(cov1.loc[clstrs[i], clstrs[i]]).flatten()
        
    cov2 = wIntra.T.dot(np.dot(cov1, wIntra)) #reduced covariance matrix
    
    # code snippet 7.5 - intercluster optimal allocations
    # step 3. compute optimal intercluster allocations, usint the reduced covariance matrix
    # which is close to a diagonal matrix, so optimization problem is close to ideal case \ro =0
    wInter = pd.Series(minVarPort(cov2)).flatten(), index=co2.index)
    wAll0 = wIntra.mul(wInter, axis=1).sum(axis=1).sort_index()
    
    