import yfinance as yf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import math

#Resources:
#Random matrix theory: https://calculatedcontent.com/2019/12/03/towards-a-new-theory-of-learning-statistical-mechanics-of-deep-neural-networks/
#Review: [Book] Commented summary of Machine Learning for Asset Managers by Marcos Lopez de Prado
#https://gmarti.gitlab.io/qfin/2020/04/12/commented-summary-machine-learning-for-asset-managers.html
#Chapter 2: This chapter essentially describes an approach that Bouchaud and his crew from the CFM have 
#pioneered and refined for the past 20 years. The latest iteration of this body of work is summarized in 
#Joel Bun’s Cleaning large correlation matrices: Tools from Random Matrix Theory.
#https://www.sciencedirect.com/science/article/pii/S0370157316303337
#Condition number: https://dominus.ai/wp-content/uploads/2019/11/ML_WhitePaper_MarcoGruppo.pdf

# Excersize 2.9:
# 2. Using a series of matrix of stock returns:
#    a) Compute the covariance matrix. 
#       What is the condition number of the correlation matrix
#    b) Compute one hundretd efficient frontiers by drawing one hundred
#       alternative vectors of expected returns from a Normal distribution
#       with mean 10% and std 10%
#    c) Compute the variance of the errors agains the mean efficient frontier.
def covariance_of_OL():
    
    """ Create covariance matrix 
    >>> import numpy as np
    >>> n = 3
    >>> T = 3
    >>> S = np.array([[1,2,3],[6,4,2],[9,1,5]])
    >>> M = np.sum(S, axis=1) * 1/n
    >>> M
    array([2, 4, 5])
    >>> demeaned_M = (S.T - M).T
    >>> print(demeaned_M)
    [[-1  0  1]
     [ 2  0 -2]
     [ 4 -4  0]]
    >>> covariance = np.dot(demeaned_M.T, demeaned_M) * (1.0/(n*T))
    >>> print(covariance)
    [[ 2.33333333 -1.77777778 -0.55555556]
     [-1.77777778  1.77777778  0.        ]
     [-0.55555556  0.          0.55555556]]
    """
        
    ol = pd.read_csv('ol_ticker.csv', sep='\t', header=None)
    ticker_name = ol[0]
    S = np.empty([10, 30])
    covariance_matrix = np.empty([10, 30])
    n = 10 # num stocks in portfolio
    T=30
    portfolio_name = [ [ None ] for x in range( 10 ) ]
    mean_stonks = np.empty([10])
    for i in range(1, n+1):  #len(ticker_name)):  # 46
        ticker = ticker_name[i]
        print(ticker)
        ol_ticker = ticker + '.ol'
        df = yf.Ticker(ol_ticker)
        #'shortName' in df.info and
        if len(df.history(period="max")) > T:  # only read tickers with more than 30 days history
            ticker_df = df.history(period="max")
            #1.Stock Data
            S[i-1] = ticker_df['Close'][-30:].values
            portfolio_name[i-1] = ol_ticker
        else:
            print("no data for ticker:" + ticker)
    
        #2.Average Price Of Stock
        mean_stonks= np.sum(S, axis=1)/T #sum along row
        #3.Demeaning The Prices
        de_meaned_S = (S.T - mean_stonks).T
        #4.Covariance Matrix
        #Once we have the de-meaned price series, we establish the
        #covariance of different stocks by multiplying the transpose of
        #the de-meaned price series with itself and divide it by 'm'
        covariance = (np.dot(de_meaned_S.T, de_meaned_S))/(n*T)
        # The eigen-values of the covariance matrix is distributed like Marcenko-Pasture dist.
        #any any eigenvalues outside distribution is signal else noise.
        
        #Standard Model: Markowitz’ Curse
        #The condition number of a covariance, correlation (or normal, thus diagonalizable) matrix is the absolute
        #value of the ratio between its maximal and minimal (by moduli) eigenvalues. This number is lowest for a diagonal
        #correlation matrix, which is its own inverse.        
        eigenvalue, eigenvector = np.linalg.eig(covariance)
        eigenvalue = abs(eigenvalue)
        condition_num = max(eigenvalue) - min(eigenvalue)


if __name__ == '__main__':
    covariance_of_OL()
    
    import doctest
    doctest.testmod()