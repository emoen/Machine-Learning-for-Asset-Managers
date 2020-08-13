# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
#import seaborn as sns
import statsmodels.api as sm1

#Code snippet 6.1 generating a set of informative, redundant, and noisy explanatory variables
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

if __name__ == '__main__':    
X, y = getTestData(40, 5, 30, 10000, sigmaStd=.1)
ols = sm1.Logit(y, X).fit()
ols.summary()
plot_data = ols.pvalues.sort_values(ascending=False)
plot_data.plot(kind='barh', figsize=(20,10), title="Figure 6.1 p-Values computed on a set of explanatory variables")
plt.show()