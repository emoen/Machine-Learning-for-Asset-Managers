# -*- coding: utf-8 -*-
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import math
import matplotlib.pylab as plt

import marcenko_pastur_pdf as mp
import chap2_monte_carlo_experiment as mc

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
def get_OL_tickers_close(T=936, N=234):       
    # N - num stocks in portfolio, T lookback time
    ol = pd.read_csv('csv/ol_ticker.csv', sep='\t', header=None)
    ticker_names = ol[0]
    S = np.empty([T, N])
    covariance_matrix = np.empty([T, N])
    portfolio_name = [ [ None ] for x in range( N ) ]
    ticker_adder = 0
    for i in range(0, len(ticker_names)):  #len(ticker_names)):  # 46
        ticker = ticker_names[i]
        print(ticker)
        ol_ticker = ticker + '.ol'
        df = yf.Ticker(ol_ticker)
        #'shortName' in df.info and
        try:
            ticker_df = df.history(period="7y")
            if ticker=='EMAS': print("****EMAS******")
            if ticker=='AVM': print("****AVM*********")
            if ticker_df.shape[0] > T and ticker!='EMAS' and ticker != 'AVM':  # only read tickers with more than 30 days history
                #1.Stock Data
                S[:,ticker_adder] = ticker_df['Close'][-T:].values # inserted from oldest tick to newest tick
                portfolio_name[ticker_adder] = ol_ticker
                ticker_adder += 1
            else:
                print("no data for ticker:" + ol_ticker)
        except ValueError:
            print("no history:"+ol_ticker)
    
    return S, portfolio_name
    
def denoise_OL(S, do_plot=True):
    
    np.argwhere( np.isnan(S) )
    
    # cor.shape = (1000,1000). If rowvar=1 - row represents a var, with observations in the columns.
    cor = np.corrcoef(S, rowvar=0) 
    eVal0 , eVec0 = mp.getPCA( cor ) 
    print(np.argwhere(np.isnan(np.diag(eVal0))))
        
    # code snippet 2.4 
    q = float(S.shape[0])/S.shape[1]#T/N
    eMax0, var0 = mp.findMaxEval(np.diag(eVal0), q, bWidth=.01)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    
    if do_plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(np.diag(eVal0), bins=100, normed = True)  #normed = True, 
        
        pdf0 = mp.mpPDF(var0, q=S.shape[0]/float(S.shape[1]), pts=N)
        pdf1 = mp.fitKDE( np.diag(eVal0), bWidth=.005) #empirical pdf
        
        #plt.plot(pdf1.keys(), pdf1, color='g') #no point in drawing this
        plt.plot(pdf0.keys(), pdf0, color='r')
        plt.show()
    
    # code snippet 2.5 - denoising by constant residual eigenvalue
    corr1 = mp.denoisedCorr(eVal0, eVec0, nFacts0)
    eVal1, eVec1 = mp.getPCA(corr1)
    
    return eVal0, eVec0, eVal1, eVec1, corr1, var0

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def calculate_correlation(S, T=936, N=234):
    """ Create covariance matrix 
    >>> import numpy as np
    >>> n = 3
    >>> T = 3
    >>> S = np.array([[1,2,3],[6,4,2],[9,1,5]])
    >>> M = np.mean(S, axis=1) # mean of row (over all T (columns))
    >>> M
    array([2, 4, 5])
    >>> demeaned_S = S - M[:,None]
    >>> print(demeaned_S)
    [[-1  0  1]
     [ 2  0 -2]
     [ 4 -4  0]]
    >>> demeaned_S= demeaned_S.astype('float32')
    >>> covariance = np.dot(demeaned_S, demeaned_S.T) * (1.0/(n-1))
    >>> print(covariance)
    [[ 1. -2. -2.]
     [-2.  4.  4.]
     [-2.  4. 16.]]
    >>> np.testing.assert_array_equal(covariance, np.cov(S))
    >>> stds = np.std(S, axis=1, ddof=1)
    >>> stds_m = np.outer(stds, stds)
    >>> covariance = covariance.astype('float32')
    >>> correlation = np.divide(covariance, stds_m)
    >>> np.testing.assert_array_equal(correlation, np.corrcoef(S))
    >>> print(correlation)
    >>> print(correlation_from_covariance(covariance))
    """

    #2.Average Price Of Stock
    M = np.sum(S, axis=1)/T #sum along row
    #3.Demeaning The Prices
    de_meaned_S = S - M[:,None]
    #4.Covariance Matrix
    #Once we have the de-meaned price series, we establish the
    #covariance of different stocks by multiplying the transpose of
    #the de-meaned price series with itself and divide it by 'm'
    covariance = (np.dot(de_meaned_S, de_meaned_S.T))/(N-1)
    # The eigen-values of the covariance matrix is distributed like Marcenko-Pasture dist.
    #any any eigenvalues outside distribution is signal else noise.
    
    #Standard Model: Markowitz’ Curse
    #The condition number of a covariance, correlation (or normal, thus diagonalizable) matrix is the absolute
    #value of the ratio between its maximal and minimal (by moduli) eigenvalues. This number is lowest for a diagonal
    #correlation matrix, which is its own inverse.        
    corr = correlation_from_covariance(covariance)
    eigenvalue, eigenvector = np.linalg.eig(np.corrcoef(S))
    eigenvalue = abs(eigenvalue)
    condition_num = max(eigenvalue) - min(eigenvalue)

#consider using log-returns
def calculate_returns( S, percentageAsProduct=False ):
    ret = np.zeros((S.shape[0]-1, S.shape[1]))
    cum_sums = np.zeros(S.shape[1])
    for j in range(0, S.shape[1]):
        cum_return = 0
        S_ret = np.zeros(S.shape[0]-1)
        for i in range(0,S.shape[0]-1):
            if percentageAsProduct==True:
                S_ret[i] = 1+((S[i+1,j]-S[i,j])/S[i,j])
            else:
                S_ret[i] = ((S[i+1,j]-S[i,j])/S[i,j])
        
        cum_return = np.prod(S_ret)-1    
        
        cum_sums[j] = cum_return
        ret[:, j] = S_ret
    
    return ret, cum_sums
    
if __name__ == '__main__':
    N = 234 #3
    T = 936
    S_value = np.loadtxt('csv/ol184.csv', delimiter=',')
    portfolio_name = pd.read_csv('csv/ol_names.csv', delimiter=',',header=None)[0].tolist()
    S_value = S_value[:,1:184] # S = S[:,6:9]
    portfolio_name = portfolio_name[1:184] #portfolio_name = portfolio_name[6:9]
    if S_value.shape[0] <1:
        S_value, portfolio_name = get_OL_tickers_close()
        np.savetxt('csv/ol184.csv', S_value, delimiter=',')
        np.savetxt('csv/ol_names.csv', np.asarray(portfolio_name), delimiter=',', fmt='%s')
        
    # use matrix of returns to calc correlation
    S, instrument_returns = calculate_returns(S_value)
    _, instrument_returns = calculate_returns(S_value, percentageAsProduct=True)
    S = S_value
    #print performance ascending    
    print(np.asarray(portfolio_name)[np.argsort(instrument_returns)])
        
    #calculate_correlation(S)
    eVal0, eVec0, denoised_eVal, denoised_eVec, denoised_corr, var0 = denoise_OL(S)
    detoned_corr = mp.detoned_corr(denoised_corr, denoised_eVal, denoised_eVec, market_component=1)
    detoned_eVal, detoned_eVec = mp.getPCA(detoned_corr)

    denoised_eigenvalue = np.diag(denoised_eVal)
    eigenvalue_prior = np.diag(eVal0)
    plt.plot(range(0, len(denoised_eigenvalue)), np.log(denoised_eigenvalue), color='r', label="Denoised eigen-function")
    plt.plot(range(0, len(eigenvalue_prior)), np.log(eigenvalue_prior), color='g', label="Original eigen-function")
    plt.xlabel("Eigenvalue number")
    plt.ylabel("Eigenvalue (log-scale)")
    plt.legend(loc="upper right")
    plt.show()
    
    #from code snippet 2.10
    detoned_cov = mc.corr2cov(detoned_corr, var0)
    w = mc.optPort(detoned_cov)
    print(w)
    #min_var_port = 1./nTrials*(np.sum(w, axis=0)) 
    #print(min_var_port)
    
    #expected portfolio variance: W^T.(Cov).W
    #https://blog.quantinsti.com/calculating-covariance-matrix-portfolio-variance/
    minVarPortfolio_var = np.dot(np.dot(w.T, detnoed_corr), w)
    
    #Expected return: w.T . mu  
    # https://www.mn.uio.no/math/english/research/projects/focustat/publications_2/shatthik_barua_master2017.pdf p8
    # or I.T.cov^-1.mu / I.T.cov^-1.I
    inv = np.linalg.inv(cov)
    e_r = np.dot(np.dot(ones.T, inv), mu) / np.dot(ones.T, np.dot(ones.T, inv))
    
    #Chapter 4 optimal clustering
        # recreate fig 4.1 colormap of random block correlation matrix
    nCols, minBlockSize = 183, 2
    print("minBlockSize"+str(minBlockSize))
    corr0 = detoned_corr

    matplotlib.pyplot.matshow(corr0) #invert y-axis to get origo at lower left corner
    matplotlib.pyplot.gca().xaxis.tick_bottom()
    matplotlib.pyplot.gca().invert_yaxis()
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.show()
    
    import doctest
    doctest.testmod()