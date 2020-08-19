# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection._split import KFold
import numpy as np
import pandas as pd
#import seaborn as sns
import statsmodels.api as sm1

#Code snippet 6.1 generating a set of informative, redundant, and noisy explanatory variables
# returns matrix X of training samples, and vector y of class labels for the training samples
def getTestData(n_features=100, n_informative=25, n_redundant=25, n_samples=10000, random_state=0, sigmaStd=.0):
    #generate a random dataset for classification problem
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features-n_redundant, 
        n_informative=n_informative, n_redundant=0, shuffle=False, random_state=random_state)
    cols = ['I_'+str(i) for i in range(0, n_informative)]
    cols += ['N_'+str(i) for i in range(0, n_features - n_informative - n_redundant)]
    X, y = pd.DataFrame(X, columns=cols), pd.Series(y)
    i = np.random.choice(range(0, n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X['R_'+str(k)] = X['I_' + str(j)] + np.random.normal(size=X.shape[0])*sigmaStd    
    return X, y

#code snippet 6.2 implementation of an ensembke MDI method
def featImpMDI(fit, featNames):
    #feat importance based on IS mean impurity reduction
    df0 = {i:tree.feature_importances_ for i, tree in enumerate(fit.enumerators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) #because max_features=1
    imp = pd.concat({'mean':df0.mean(), 'std':df0.std()*df0.shape[0]**-.5}, axis=1) #CLT
    imp /= imp['mean'].sum()
    return imp
    
#code snippet 6.3 implementation of MDA - MDI with cross validation
def featImpMDA(clf, X, y, n_splits=10):
    #feat importance based on OOS score reduction
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1=pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        x0, y0 = X.iloc[train, :], y.iloc[train]
        x1, y1 = X.iloc[test,:], y.iloc[test]
        fit = clf.fit(X=X0, y=y0) # the fit occures
        prob= fit.predict_proba(X1) #prediction before shuffles
        scr0.loc[i]=-log_loss(y1, prob, labelslf.classes_)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) #shuffle one columns
            prob = fit.predict_proba(X1_) #prediction after shuffle
            scr1.loc[i,j] = -log_loss(y1, prob, labels=clf.classes_)
    imp=(-1*scr1).add(scr0, axis=0)
    imp = imp/(-1*scr1)
    imp=pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-.5}, axis=1) #CLT
    return imp

if __name__ == '__main__':    
    X, y = getTestData(40, 5, 30, 10000, sigmaStd=.1)
    ols = sm1.Logit(y, X).fit()
    ols.summary()
    plot_data = ols.pvalues.sort_values(ascending=False)
    plot_data.plot(kind='barh', figsize=(20,10), title="Figure 6.1 p-Values computed on a set of explanatory variables")
    plt.show()
    
    #code snippet 6.2
    X, y = getTestData(40, 4, 30, 10000, sigmaStd=.1)
    clf = DecisionTreeClassifier(criterion='entropy', 
                                 max_features=1, 
                                 class_weight='balanced', 
                                 min_weight_fraction_leaf=0)
                                 
    clf = BaggingClassifier(base_estimator=clf, 
                          n_estimators=1000, 
                          max_features=1., 
                          max_samples=1., 
                          oob_score=False)
    fit = clf.fit(X,y)
    imp = featImpMDI(fit, featNames=X.columns)
    
    #print the graph Example 6.2 Example of MDI results
    imp.sort_values('mean', inplace=True)
    plt.figure(figsize=(10, imp.shape[0] / 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('Figure 6.2 Example of MDI results')
    plt.show()