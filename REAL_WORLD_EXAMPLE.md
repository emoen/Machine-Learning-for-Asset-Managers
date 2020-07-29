

# Machine-Learning-for-Asset-Managers
Lets apply the lessons from the book to real world data using the data from the last 936 trading days from Oslo Børs. That means we have 183 instruments

## Chapter 2 Denoising and Detoning

S_value contains the time-series of instruments. One instrument per column.
```
>>> S_value
array([[  7.3311,  94.83  ,   3.4349, ...,  90.03  , 234.85  ,  27.6   ],
       [  7.1928,  94.83  ,   3.4139, ...,  90.43  , 238.69  ,  29.15  ],
       [  7.1005,  94.83  ,   3.4069, ...,  89.63  , 239.58  ,  29.15  ],
       ...,
       [  1.36  , 112.    ,   4.22  , ...,  21.6   , 337.7   ,  34.    ],
       [  1.465 , 112.    ,   4.315 , ...,  21.62  , 334.    ,  33.7   ],
       [  1.38  , 112.    ,   4.33  , ...,  22.48  , 331.2   ,  33.5   ]])
>>> S_value.shape
(936L, 183L)
```
There is a corresponding list of portfolio names:
```
>>> portfolio_name[0:3]
['FIVEPG.ol', 'AASB-ME.ol', 'ASC.ol']
>>> portfolio_name[-3:]
['XXL.ol', 'YAR.ol', 'ZAL.ol']
```
The time-series of values must be converted to returns.
```
>>> S, instrument_returns = calculate_returns(S_value)
>>> S
array([[-0.01886484,  0.        , -0.00611372, ...,  0.00444296,
         0.01635086,  0.05615942],
       [-0.01283228,  0.        , -0.00205044, ..., -0.00884662,
         0.00372869,  0.        ],
       [ 0.        ,  0.        ,  0.00616396, ...,  0.00892558,
         0.00784707, -0.01989708],
       ...,
       [-0.13375796, -0.00884956,  0.0047619 , ..., -0.02262443,
        -0.0088054 ,  0.        ],
       [ 0.07720588,  0.        ,  0.02251185, ...,  0.00092593,
        -0.01095647, -0.00882353],
       [-0.05802048,  0.        ,  0.00347625, ...,  0.03977798,
        -0.00838323, -0.00593472]])
```
Fitting the covariance matrix to the marcenko-pasture distribution assumes the elements of the matrix is normal distribution with mean 0 and variance:
```
>>>eVal0, eVec0, denoised_eVal, denoised_eVec, denoised_corr, var0 = denoise_OL(S)
>>> var0
0.8858105886313286
```
Fitting denoised_corr to the marcenko-pasture pdf:
| ![marcenko-pastur of oslo børs last 936 trading days](https://github.com/emoen/Machine-Learning-for-Asset-Managers/blob/master/ol_n_183_T_936.png) | 
|:--:| 
| *Marcenko-Pasture theoretical probability density function, and empirical density function from 183 instruments the last 936 trading days:* |

The largest eigenvalues of 24.02 is the market and can be de-toned

