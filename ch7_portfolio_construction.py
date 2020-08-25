import numpy as np
import matplotlib.pyplot as mp1
import seaborn as sns

import ch2_monte_carlo_experiment as mc


# code snippet 7.1 - Composition of block-diagonal correlation matrices
if __name__ == '__main__': 
    corr0 = mc.formBlockMatrix(2, 2, .5)
    eval, eVec = np.linalg.eigh(corr0)
    print(max(eVal)/min(eVal))
    sns.heatmap(corr0, cmap='viridis')