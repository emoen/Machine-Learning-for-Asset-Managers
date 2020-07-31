# Machine-Learning-for-Asset-Managers

Implementation of code snippets and exercises from [Machine Learning for Asset Managers (Elements in Quantitative Finance)](https://www.amazon.com/Machine-Learning-Managers-Elements-Quantitative/dp/1108792898)
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

* definition of a metric: 
   1. identity of indiscernibles d(x,y) = 0 => x=y 
   2. Symmetry d(x,y) = d(y,x) 
   3. triangle inequality. 
   - 1,2,3 => non-negativ, d(x,y) >= 0 
* pearson correlation
* distance correlation
* angular distance
* Information-theoretic codependence/entropy dependence
    - cross-entropy:  H[X] = - &Sigma;<sub>s &isin; S<sub>X</sub></sub> p[x] log (p[x])
    - Kullback-Leilbler divergence:  D<sub>KL</sub>[p||q] = - &Sigma;<sub>s &isin; S<sub>X</sub></sub> p[x] log (q[x]/p[x]) = p[x] &Sigma;<sub>s &isin; S</sub> log (p[x]/q[x])
    - Cross-entropy: H<sub>c</sub>[p||q] = H[x] = D<sub>KL</sub>[p||q]
    - Mutual information: Decrease in uncertainty in X from knowing Y: I[X,Y] = H[X] - H[X|Y] = H[X] + H[Y] - H[X,Y] = E<sub>X</sub>[D<sub>KL</sub>[p[y|x]||p[y]]]
    - variation of information: VI[X,Y] = H[X|Y] + H[Y|X] = H[X,Y] - I[X,Y]. It is uncertainty we expect in one variable given another variable: VI[X,Y] = 0 <=> X=Y
    - Kullback-Leilbler divergence is not a metric while variation of information is.
   
 
 ```
 >>> ss.entropy([1./2,1./2], base=2)
1.0
>>> ss.entropy([1,0], base=2)
0.0
>>> ss.entropy([1./3,2./3], base=2)
0.9182958340544894
```
1 bit of information in coin toss. 0 bit of information in deterministic outcome, less than 1 bit of information in unfair coin toss.  

## Chapter 4 Optimal Clustering

Use unsupervised learning to maximize intragroup similarities and minimize intergroup similarities. Consider matrix X of shape N x F. N objects and F features. Features are used to compute proximity(correlation, mutual information) to N objects in an NxN matrix.

There are 2 types of clustering algorithms. Partitional and hierarchical:
1. Connectivity: hierarchical clustering
2. Centroids: like k-means
3. Distribution: gaussians
4. Density: search for connected dense regions like DBSCAN, OPTICS
5. Subspace: modeled on two dimension, feature and observation. [example](https://quantdare.com/biclustering-time-series/)
    
