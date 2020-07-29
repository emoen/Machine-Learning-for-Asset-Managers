# Machine-Learning-for-Asset-Managers

Implementation of code snippets and excersizes from [Machine Learning for Asset Managers (Elements in Quantitative Finance)](https://www.amazon.com/Machine-Learning-Managers-Elements-Quantitative/dp/1108792898)
written by Prof. Marcos López de Prado.

The project is for my own learning. If you want to use the consepts from the book - you should head over to Hudson & Thames. They have implemented these consepts and many more in [mlfinlab](https://github.com/hudson-and-thames/mlfinlab)

## Chapter 2 Denoising and Detoning

Marcenko-Pasture theoretical probability density function, and empirical density function:
| ![marcenko-pastur](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/gaussian_mp.png) | 
|:--:| 
| *Marcenko-Pasture theoretical probability density function, and empirical density function:* |


Denoising a random matrix with signal using the constant residual eigenvalue method. This is done by fixing random eigenvalues. See code snippet 2.5
| ![eigenvalue_method.jpg](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/figure_2_3_eigenvalue_method.png) | 
|:--:| 
| *A comparison of eigenvalues before and after applying the residual eigenvalue method:* |

Detoned covariance matrix can be used to calculate minimum variance portfolio. The efficient frontier is the upper portion of the minimum variance frontier starting at the minimum variance portfolio. A denoised covariance matrix is less unstable to change.

## Chapter 3 Distance Metrics
