# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.model_selection import LeaveOneOut

'''
Selecting the bandwidth via cross-validation
The choice of bandwidth within KDE is extremely important to finding a suitable density estimate, 
and is the knob that controls the bias–variance trade-off in the estimate of density: 
too narrow a bandwidth leads to a high-variance estimate (i.e., over-fitting), 
where the presence or absence of a single point makes a large difference. 
Too wide a bandwidth leads to a high-bias estimate (i.e., under-fitting) 
where the structure in the data is washed out by the wide kernel.

There is a long history in statistics of methods to quickly estimate the best bandwidth 
based on rather stringent assumptions about the data: if you look up the KDE implementations 
in the SciPy and StatsModels packages, for example, you will see implementations 
based on some of these rules.

In machine learning contexts, we've seen that such hyperparameter tuning often is 
done empirically via a cross-validation approach. With this in mind, the KernelDensity estimator 
in Scikit-Learn is designed such that it can be used directly within the Scikit-Learn's standard 
grid search tools. Here we will use GridSearchCV to optimize the 
bandwidth for the preceding dataset. Because we are looking at such a small dataset, 
we will use leave-one-out cross-validation, which minimizes the reduction in 
training set size for each cross-validation trial:

https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
'''

''' This is also excersize 2.7 in the book:
"Extend function fitKDE in code snippet 2.2, so that it estimates through
cross-validation the optimal value of bWidth (bandwidth)"
'''

def findOptimalBWidth(eigenvalues):
    bandwidths = 10 ** np.linspace(-2, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(eigenvalues[:, None]);
    
    #Now we can find the choice of bandwidth which maximizes the score (which in this case defaults to the log-likelihood):
    
    return grid.best_params_