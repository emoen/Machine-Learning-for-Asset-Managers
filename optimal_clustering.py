import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

#codesnippet 4.1
#base clustering
def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    x, silh= ((1-corr0.fillna(0))/2.)**.5, pd.Series() #observations matrix
    for init in range(n_init):
        for i in range(2, maxNumClusters+1):
            kmeans_ = KMeans(n_clusters=i, n_jobs=1, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            if np.isnan(stat[l]) or stat[0]>stat[1]:
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
    if maxNumClusters == None:maxNumClusters=corr0.shape[1]-1
    corr1, clstrs, silh=clusterKMeansBase(corr0, maxNumClusters=min(maxNumClusters, corr0.shape[1]-1), n_init=n_init)
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
       
        
    