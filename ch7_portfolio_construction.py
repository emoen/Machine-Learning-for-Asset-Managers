import numpy as np
import matplotlib.pyplot as mp1
import seaborn as sns

import ch2_monte_carlo_experiment as mc


# code snippet 7.1 - Composition of block-diagonal correlation matrices
if __name__ == '__main__': 
    corr0 = mc.formBlockMatrix(2, 2, .5)
    eVal, eVec = np.linalg.eigh(corr0)
    matrix_condition_number = max(eVal)/min(eVal)
    print(matrix_condition_number) 
    
    fig, ax = plt.subplots(figsize=(13,10))  
    sns.heatmap(corr0, cmap='viridis')
    plt.show()