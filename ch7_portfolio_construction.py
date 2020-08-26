import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import matplotlib.pyplot as mp1
import seaborn as sns

import ch2_monte_carlo_experiment as mc
import ch2_marcenko_pastur_pdf as mp
import ch4_optimal_clustering as oc


es
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
    np.random.seed(0)
    mu0, cov0 = mc.formTrueMatrix(nBlocks, bSize, bCorr)
    cols = cov0.columns
    cov1 = mc.deNoiseCov(cov0, q, bWidth=.01) #denoise cov
    cov1 = pd.DataFrame(cov1, index=cols, columns=cols)
    corr1 = mp.cov2corr(cov1)
    corr1, clstrs, silh = oc.clusterKMeansBase(pd.DataFrame(corr0))
    
    