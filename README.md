# Machine-Learning-for-Asset-Managers

Implementation of code snippets and exercises from [Machine Learning for Asset Managers (Elements in Quantitative Finance)](https://www.amazon.com/Machine-Learning-Managers-Elements-Quantitative/dp/1108792898)
written by Prof. Marcos López de Prado.

The project is for my own learning. If you want to use the consepts from the book - you should head over to Hudson & Thames. They have implemented these consepts and many more in [mlfinlab](https://github.com/hudson-and-thames/mlfinlab). Edit: seems like some of theyr work - like jupyter notebooks - has gone behind a paywall - doh.

## Chapter 2 Denoising and Detoning

Marcenko-Pasture theoretical probability density function, and empirical density function:
| ![marcenko-pastur.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/gaussian_mp.png) | 
|:--:| 
| *Marcenko-Pasture theoretical probability density function, and empirical density function:* |


Denoising a random matrix with signal using the constant residual eigenvalue method. This is done by fixing random eigenvalues. See code snippet 2.5
| ![eigenvalue_method.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/figure_2_3_eigenvalue_method.png) | 
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
1. 1 bit of information in coin toss
2. 0 bit of information in deterministic outcome
3. less than 1 bit of information in unfair coin toss


* Angular distance: p_d = sqrt(1/2 - (1-rho(X, Y)))
* Absolute angular distance: p_d = sqrt(1/2 - (1-|rho|(X, Y)))
* Squared angular distance: p_d = sqrt(1/2 - (1-rho^2(X, Y)))

![fig_3_1_angular_distance.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_3_1_angular_distance.png)  ![fig_3_1_abs_squared_angular_distance.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_3_1_abs_squared_angular_distance.png) 
Standard angular distance is better used for long-only portfolio appliacations. Squared and Absolute Angular Distances for long-short portfolios. 

## Chapter 4 Optimal Clustering

Use unsupervised learning to maximize intragroup similarities and minimize intergroup similarities. Consider matrix X of shape N x F. N objects and F features. Features are used to compute proximity(correlation, mutual information) to N objects in an NxN matrix.

There are 2 types of clustering algorithms. Partitional and hierarchical:
1. Connectivity: hierarchical clustering
2. Centroids: like k-means
3. Distribution: gaussians
4. Density: search for connected dense regions like DBSCAN, OPTICS
5. Subspace: modeled on two dimension, feature and observation. [Example](https://quantdare.com/biclustering-time-series/)


Generating of random block correlation matrices is used to simulate instruments with correlation. The utility for doing this is in code snippet 4.3, and it uses clustering algorithms <i>optimal number of cluster</i> (ONC) defined in snippet 4.1 and 4.2, which does not need a predefined number of clusters (unlike k-means), but uses an 'elbow method' to stop adding clusters. The optimal number of clusters are achived when there is high intra-cluster correlation and low inter-cluster correlation. The [silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)) is used to minimize within-group distance and maximize between-group distance. 
| ![random_block_corr_matrix.jpg](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_4_1_random_block_correlation_matrix_mini.png) | 
|:--:| 
| *Random block correlation matrix. Light colors indicate a high correlation, and dark colors indicate a low correlation. In this example, the number of blocks K=6, minBlockSize=2, and number of instruments N=30* |
| ![fig_4_1_random_block_correlation_matrix_onc.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_4_1_random_block_correlation_matrix_onc_mini.png) | 
| *Applying the ONC algorithm to the random block correlation matrix. ONC finds all the clusters.* |

## Chapter 5 Financial Labels

* Fixed-Horizon method
* Time-bar method
* Volume-bar method

Tiple-Barrier Method involves holding a possition until
1. Unrealized profit target achieved
2. unrealized loss limit reached
3. Position is held beyond a maximum number of bars

Trend-scanning method: the idea is to identify trends and let them run for as long and as far as they may persists, without setting any barriers. 

| ![fig_5_1_trend_scanning.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_5_1_trend_scanning.png) | 
|:--:| 
| *Example of trend-scanning labels on sine wave with gaussian noise:* |

| ![fig_5_2_trend_scanning_t_values.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_5_2_trend_scanning_t_values.png) | 
|:--:| 
| *trend-scanning with t-values which shows confidence in trend. 1 is high confidence going up and -1 is high confidence going down.* |

## Chapter 6 Feature Importance Analysis

<i>"p-value does not measure the probability that neither the null nor the alternative hypothesis is true, or the significance of a result."</i>
| ![fig_6_1_p_values_explanatory_vars.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_6_1_p_values_explanatory_vars.png) | 
|:--:| 
| *p-Values computed on a set of informative, redundant, and noisy explanatory variables. The explanatory variables has not the hightest p-values.* |

The MDI algorith deals with 3 out of 4 problems with p-values:
1. MDI is not imposing any tree structure, algebraic specification, or relying on any stocastic or distributional characteristics of the residuals (e.g. y=b<sub>0</sub>+b<sub>1</sub>*x<sub>i</sub>+&epsilon;)
2. betas are estimated from single sample, MDI relies on bootstrapping, so the variance can be reduced by the numbers of trees in the random forrest ensemble.
3. In MDI the goal is not to estimate a coefficient of a given algebraic equation (b_hat_0, b_hat_1) describing the probability of a null-hypotheses.
4. MDI does not correct of calculation in-sample, as there is no cross-validation.

| ![fig_6_2_mdi_example.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_6_2_mdi_example.png) | 
|:--:| 
| *MDI algorithm example* |

Figure 6.4 shows that ONC correctly recognizes that there are six relevant clusters(one cluster for each informative feature, plus one cluster of noise features), and it assigns the redundant features to the cluster that contains the informative feature from which the redundant features where derived. Given the low correlation across clusters, there is no need to replace the features with their residuals.
| ![fig_6_4_feature_clustering.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_6_4_feature_clustering.png) | 
|:--:| 

Next, apply the clustered MDI method to the clustered data:
| ![fig_6_5_clustered_MDI.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_6_5_clustered_MDI.png) | 
|:--:| 
| *Figure 6.5 Clustered MDI* |

Clustered MDI works better han non-clustered MDI. Finally, apply the clustered MDA method to this data:
| ![fig_6_6_clustered_MDA.png](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/img/fig_6_6_clustered_MDA.png) | 
|:--:| 
| *Figure 6.6 Clustered MDA* |

Conclusion: C_5 which is accosiated with noisy features is not important, and all other clusteres has similar importance.

## Chapter 7 Portfolio Construction

Convex portfolio optimization can calculate minimum variance portfolio and max sharp-ratio.

def Condition number: absolute value of the ratio between the maximum and minimum eigenvalues: A_n_n / A_m_m. The condition number says something about the instability of the instability caused by covariance structures.
def trace = sum(diag(A)) - its the sum of the diagonal elements

Highly correlated time-series implie high condition number of the correlation matrix.
