# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.linalg import block_diag

import marcenko_pastur_pdf as mp

# Code snippet 2.7
#Generate a block-diagnoal covariance matrix and a vector of means
def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones( (bSize, bSize))*bCorr
    block[range(bSize), range(bSize)] = 1
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
    cov = corr*np.outer(std, std)
    return cov
    
if __name__ == '__main__':
    nBlocks, bSize, bCorr = 10, 50, .5
    np.random.seed(0)
    mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)
    