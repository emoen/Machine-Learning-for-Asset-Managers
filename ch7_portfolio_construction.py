import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as mp1
import seaborn as sns

import ch2_monte_carlo_experiment as mc


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
    