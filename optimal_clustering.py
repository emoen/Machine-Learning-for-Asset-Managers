import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_random_state
from scipy.linalg import block_diag

import marcenko_pastur_pdf as mp
import matplotlib.pylab as plt
import matplotlib

#codesnippet 4.1
#base clustering
def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    x, silh = ((1-corr0.fillna(0))/2.)**.5, pd.Series() #observations matrix
    maxNumClusters = min(maxNumClusters, x.shape[0]-1)
    for init in range(n_init):
        for i in range(2, maxNumClusters+1):
            print(i)
            kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh, kmeans = silh_, kmeans_
    
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx] #reorder rows
    
    corr1 = corr1.iloc[:, newIdx] #reorder columns
    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} #cluster members
    silh = pd.Series(silh, index=x.index)
    
    return corr1, clstrs, silh
    
#codesnippet 4.2
#Top level of clustering
def makeNewOutputs(corr0, clstrs, clstrs2):
    clstrsNew={}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])
    
    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx, newIdx]
    x = ((1-corr0.fillna(0))/2.)**.5
    kmeans_labels = np.zeros(len(x.columns))
    for i in clstrsNew.keys()):
        idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs]=i
    
    silhNew = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    
    return corrNew, clstrsNew, silhNew

def clusterKMeansTop(corr0, maxNumClusters=None, n_init=10):
    if maxNumClusters == None:
        maxNumClusters = corr0.shape[1]-1
        
    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters=min(maxNumClusters, corr0.shape[1]-1), n_init=n_init)
    clusterTstats={i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = sum(clusterTstats.values())/len(clsterTstats)
    redoClusters=[i for i in clusterTstats.keys() if clusterTstats[i]<tStatMean]
    if len(redoClusters)<=1:
        return corr1, clstrs, silh
    else:
        keysRedo= [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo, keysRedo]
        tStatMean=np.mean([clusterTstats[i] for i in redoClusters])
        corr2, clstrs2, silh2=clusterKmeansTop(corrTmp, maxNumClusters=min(maxNumClusters, corrTmp,.shape[1]-1),n_init=n_init)
        
#codesnippet 4.3 - utility for monte-carlo simulation
# Random block correlation matrix creation
# Simulates a time-series of atleast 100 elements. 
# So each column is highly correlated for small sigma and less correlated for large sigma (standard deviation)
#
# two matrixes of N(0,sigma^2) rv added which results in variance=2*sigma^2
def getCovSub(nObs, nCols, sigma, random_state=None):
    #sub correl matrix
    rng = check_random_state(random_state)
    if nCols == 1:
        return np.ones((1,1))
    ar0 = rng.normal(size=(nObs, 1)) #array of normal rv
    ar0 = np.repeat(ar0, nCols, axis=1) #matrix of columns repeating rv. Simulate time-series of at least 100 elements.
    ar0 += rng.normal(loc=0, scale=sigma, size=ar0.shape) #add two rv X~Y~N(0,1), Z=X+Y~N(0+0, 1+1)=N(0,2)
    ar0 = np.cov(ar0, rowvar=False) #ar0.shape = nCols x nCols
    return ar0

#generate a block random correlation matrix
#
# The last block in the matrix is going to be as large as possible
# Controlling the size of the last block matrix can be done by inceasing minBlockSize
# 
# parts is the size of the blocks. If nCols, nBlocks, minBlockSize = 6,3,1
# then parts = [1,1,4] resulting in 1x1, 1x1, 4x4 block-covariance-matrixes
# If block > 1x1 matrix the diagonal is 2 or 2*sigma as the variance of
# covariance from getCovSub() is Z=X+Y => 2*sigma
def getRndBlockCov(nCols, nBlocks, minBlockSize=1, sigma=1., random_state=None):
    
    print("getRndBlockCov:"+str(minBlockSize))
    rng = check_random_state(random_state)
    parts = rng.choice(range(1, nCols-(minBlockSize-1)*nBlocks), nBlocks-1, replace=False)
    parts.sort()
    parts = np.append(parts, nCols-(minBlockSize-1)*nBlocks) #add nCols to list of parts, unless minBlockSize>1
    parts = np.append(parts[0], np.diff(parts)-1+minBlockSize)
    print("block sizes:"+str(parts))
    cov=None
    for nCols_ in parts:
        cov_ = getCovSub(int(max(nCols_*(nCols_+1)/2., 100)), nCols_, sigma, random_state=rng)
        if cov is None:
            cov = cov_.copy()
        else: 
            cov = block_diag(cov, cov_) #list of square matrix on larger matrix on the diagonal
    
    return cov

# add two random covariance matrixes and return the correlation matrix as a dataframe. 
#
# The first covariance matrix consists of nBlocks
# and the second matrix consists of 1 block - which adds noice.
# Note: noice is also added in each block matrix. Why is noice added 2 times?
def randomBlockCorr(nCols, nBlocks, random_state=None, minBlockSize=1):
    #Form block corr
    rng = check_random_state(random_state)
    
    print("randomBlockCorr:"+str(minBlockSize))
    cov0 = getRndBlockCov(nCols, nBlocks, minBlockSize=minBlockSize, sigma=.5, random_state=rng)
    cov1 = getRndBlockCov(nCols, 1, minBlockSize=minBlockSize, sigma=1., random_state=rng) #add noise
    cov0 += cov1
    corr0 = mp.cov2corr(cov0)
    corr0 = pd.DataFrame(corr0)
    return corr0
    
if __name__ == '__main__':
    nCols, nBlocks = 6, 3
    nObs = 8
    sigma = 1.
    corr0 = randomBlockCorr(nCols, nBlocks)
    testGetCovSub = getCovSub(nObs, nCols, sigma, random_state=None) 
    tmp = testGetCovSub.flatten()
    
    # recreate fig 4.1 colormap of random block correlation matrix
nCols, nBlocks, minBlockSize = 30, 6, 2
print("minBlockSize"+str(minBlockSize))
corr0 = randomBlockCorr(nCols, nBlocks, minBlockSize=minBlockSize)
#matplotlib.pyplot.matshow(corr0)

matplotlib.pyplot.matshow(corr0)
matplotlib.pyplot.colorbar()
#ax = matplotlib.pyplot.gca()
#ax.xaxis.tick_bottom()
matplotlib.pyplot.show()




    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(tmp, bins=5, normed = True)
    plt.show()
        
    testGetRndBlockCov = getRndBlockCov(10, 3)
       
        
    