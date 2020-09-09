# -*- coding: utf-8 -*-
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import math
import matplotlib.pylab as plt
import matplotlib

import ch2_marcenko_pastur_pdf as mp
import ch2_monte_carlo_experiment as mc
import onc as onc
import ch5_financial_labels as fl
import ch7_portfolio_construction as pc

import trend_scanning as ts

import ch2_fitKDE_find_best_bandwidth as best_bandwidth

import nco as nco

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
        ax.hist(np.diag(eVal0), density = True, bins=100) #, normed = True)  #normed = True, 
        
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
    
def testNCO():
    # Chapter 7 - apply the Nested Clustered Optimization (NCO) algorithm
    N = 5
    T = 5
    S_value = np.array([[1.,2,3,4,5],
                        [1.1,3,2,3,5],
                        [1.2,4.,1.3,4,5],
                        [1.3,5,1,3,5],
                        [1.4,6,1,4,5.5],
                        [1.5,7,1,3,5.5]])
    S, instrument_returns = calculate_returns(S_value)
    _, instrument_returns = calculate_returns(S_value, percentageAsProduct=True)
    np.argsort(instrument_returns)
    #array([2, 3, 4, 0, 1], dtype=int64)
    instrument_returns
    #array([ 0.5       ,  2.5       , -0.66666667, -0.25      ,  0.1       ])
    
    eVal0, _ = getPCA(np.cov(S_value,rowvar=0, ddof=1))
    #eVal0, eVec0, denoised_eVal, denoised_eVec, denoised_corr, var0 = denoise_OL(S_value)
    q = float(S.shape[0])/float(S.shape[1])#T/N
    bWidth = best_bandwidth.findOptimalBWidth(np.diag(eVal0))
    #cov1_d = mc.deNoiseCov(np.cov(S_value,rowvar=0, ddof=1), q, bWidth['bandwidth'])
    
    mu1 = None
    cov1_d = np.cov(S,rowvar=0, ddof=1)
    min_var_markowitz = mc.optPort(cov1_d, mu1).flatten()
    min_var_NCO = pc.optPort_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()      
    mlfinlab_NCO= nco.NCO().allocate_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()

    cov1_d = np.cov(S_value,rowvar=0, ddof=1)    
    mlfinlab_NCO= nco.NCO().allocate_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()
    '''
    >>> min_var_markowitz
    array([ 1.06869282, -0.05708545,  0.03451679,  0.00133102, -0.04745517])
    >>> min_var_NCO
    array([1.18787119, 0.00808112, 0.02246385, 0.00876234, 1.54625948])
    >>> instrument_returns
    array([ 0.5       ,  2.5       , -0.66666667, -0.25      ,  0.1       ])
    >>> mlfinlab_NCO
    array([-0.27372046,  0.00308304,  0.00840394,  0.56060186,  0.70163162])
    #on S_value - so the time series
    >>> mlfinlab_NCO
    array([ 1.11111111e+00, -1.11111111e-01, -1.60963389e-17,  8.99486247e-17, -6.97001845e-17])
    '''    
    
if __name__ == '__main__':
    N = 234 #3
    T = 936
    S_value = np.loadtxt('csv/ol184.csv', delimiter=',')
    portfolio_name = pd.read_csv('csv/ol_names.csv', delimiter=',',header=None)[0].tolist()
    S_value = S_value[:,1:184] # S = S[:,6:9]
    portfolio_name = portfolio_name[1:184] #portfolio_name = portfolio_name[6:9]
    if S_value.shape[0] < 1:
        S_value, portfolio_name = get_OL_tickers_close()
        np.savetxt('csv/ol184.csv', S_value, delimiter=',')
        np.savetxt('csv/ol_names.csv', np.asarray(portfolio_name), delimiter=',', fmt='%s')
        
    # use matrix of returns to calc correlation
    S, instrument_returns = calculate_returns(S_value)
    _, instrument_returns = calculate_returns(S_value, percentageAsProduct=True)
    #S = S_value
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
    corr1, clstrs, silh = oc.clusterKMeansTop(pd.DataFrame(detoned_corr)) #1: [18, 24, 57, 81, 86, 99, 112, 120, 134, 165]
    corr11, clstrs11, silh11 = onc.get_onc_clusters(pd.DataFrame(detoned_corr)) #test with mlfinlab impl: 1: [18, 24, 57, 81, 86, 99, 112, 120, 134, 165]
    
    matplotlib.pyplot.matshow(corr11) #invert y-axis to get origo at lower left corner
    matplotlib.pyplot.gca().xaxis.tick_bottom()
    matplotlib.pyplot.gca().invert_yaxis()
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.show()
    
    #Chapter 5 Financial labels
    #Lets try trend-following on PHO
    pho = S_value[:,121]
    df0 = pd.Series(pho[-50:])
    df1 = fl.getBinsFromTrend(df0.index, df0, [3,10,1]) #[3,10,1] = range(3,10)
    tValues = df1['tVal'].values
    plt.scatter(df1.index, df0.loc[df1.index].values, c=tValues, cmap='viridis') #df1['tVal'].values, cmap='viridis')
    plt.colorbar()
    plt.show()

    bgbio_df = yf.Ticker("BGBIO.ol")
    bg_bio_ticker_df = bgbio_df.history(period="7y")
                
    bgbio = bg_bio_ticker_df['Close']
    df0 = pd.Series(bgbio[-200:])
    df1 = fl.getBinsFromTrend(df0.index, df0, [3,20,1]) #[3,10,1] = range(3,10)
    tValues = df1['tVal'].values
    plt.scatter(df1.index, df0.loc[df1.index].values, c=tValues, cmap='viridis') #df1['tVal'].values, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    S, pnames = get_OL_tickers_close()

    #get t-statistics from all instruments on OL
    S, pnames = get_OL_tickers_close(T=200,N=237)
    names = pd.DataFrame(pnames)
    names.to_csv('csv/ol237.csv', sep=',', index=False, header=False)
    np.savetxt("csv/ol_ticker237.csv", S, delimiter=',')
    
    np.argwhere(np.isnan(S))
    S[182, 110] = S[181,110]

    #implementing from book
    abc = [None for i in range(0,237)]
    for i in range(0, 20):#len(pnames)):
        instrument = S[:,i]
        df0 = pd.Series(instrument)
        print("running bins on:"+pnames[i]+" i:"+str(i))
        abc[i] = fl.getBinsFromTrend(df0.index, df0, [3,10,1])['tVal']
    
    tValLatest =  [abc[i].values[-20] for i in range(0, len(abc))]
    #most significant t-value:
    np.max(tValLatest)
    pnames[np.argmax(tValLatest)]
    #END / implementing from book
    
    #mlfinlab impl
    S[181,110]=S[180,110] #nan
    abc = [None for i in range(0,237)]
    for i in range(0, len(abc)):
        ticker_close = pd.DataFrame(S[:,i], columns={'ticker'})
        print(i)
        t_events = ticker_close.index
        tr_scan_labels = ts.trend_scanning_labels(ticker_close, t_events, 20)
        abc[i] = tr_scan_labels['t_value']
    
    abc = np.asarray(abc)
    tValLatest =  [abc[i,-20] for i in range(0, len(abc))]
    #most significant t-value:
    np.max(tValLatest)
    pnames[np.argmax(tValLatest)]
    
    plt.scatter(ticker_close.index, S[:,78], c=abc[78], cmap='viridis')

    # Chapter 7 - apply the Nested Clustered Optimization (NCO) algorithm
    N = 234 
    T = 936
    S_value = np.loadtxt('csv/ol184.csv', delimiter=',')
    S, instrument_returns = calculate_returns(S_value)
    _, instrument_returns = calculate_returns(S_value, percentageAsProduct=True)
    np.argsort(instrument_returns)
    #26,  84, 167,  35,  76, 169,  31, 137,  28,  64,  36,  37,  92, 116], dtype=int64)
    
    eVal0, eVec0, denoised_eVal, denoised_eVec, denoised_corr, var0 = denoise_OL(S)
    q = float(S.shape[0])/float(S.shape[1])#T/N
    bWidth = best_bandwidth.findOptimalBWidth(np.diag(eVal0))
    cov1_d = mc.deNoiseCov(np.cov(S,rowvar=0, ddof=1), q, bWidth['bandwidth'])
    
    mu1 = None
    min_var_markowitz = mc.optPort(cov1_d, mu1).flatten()
    min_var_NCO = pc.optPort_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()
    
    # calulate on time-series not returns
    cov1_d = np.cov(S_value,rowvar=0, ddof=1)   
    min_var_markowitz = mc.optPort(cov1_d, mu1).flatten()
    min_var_NCO = pc.optPort_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()
    #note pnames = pnames[1:] - first element is obx

    '''>>> np.argsort(min_var_markowitz)
    array([ 22,  10, 115, 151, 158, 175,  83,  23, 180, 102,  62,  57, 119,
       114,   6,  68,  33, 113,  92, 137,  96,   8, 106,  29,  76,  77,
         7, 116, 127,  20, 121,  93, 125,  72, 148, 162,  43,  99, 130,
       108, 122, 100,  66,   5, 161, 168,  94,  18,  39,  15, 150,  88,
        41,  95, 120,  52,  98, 118,  13, 105, 164, 132,  44,  53,  12,
        35, 124, 101, 107,  40, 140,  63, 111, 159,  27, 160,  59, 128,
        38,  90,  91,  11,   0, 117, 142, 146,  60,  36,  45,  65,  19,
        70, 179, 169,  78, 157,  64,  42,  89,  28,  21,  74, 123,  69,
        87, 145, 172, 182, 129, 176, 104, 166,  26, 173,   3,  80, 133,
        49,  32,  81, 178,  82,  55,  24,  56,  73,  97,  51,  79,  58,
        71,  16, 134, 163,  31,  30, 135, 141,  25,  48, 131, 171,  50,
       136,  84, 103, 126,  34, 155,  14,   9, 177, 109,  47,  46, 181,
       167, 138,   2, 153,  67, 139,  61, 110,  37, 156,  54,  86, 143,
       147, 152,   1,   4, 165, 154, 112, 144, 149,  75, 170,  17, 174,
        85], dtype=int64)
        
        >>> np.argsort(min_var_NCO)
    array([107,  91,  94,  59,  35, 172, 101,  27, 140, 150, 124,  63, 132,
       118,  53,  44,  28, 128,   0,   7, 164,  12, 117,  39,  11, 108,
       159, 168,  41, 125, 129,  78, 146,  45,  88,  52,  99, 120, 142,
        66,   5,  32,  98,  74, 169, 160,  70,  40,  77,  29, 179, 182,
       157,  90,  93,  49,  26,  60,  19,  65,  15, 100,  38,  43,  42,
        64, 162,   6,  21,  81,  69,   3, 123, 130, 178, 122,   8, 173,
       127, 148,  20, 133, 145,  72,  89,  18,  13,  82,  36,  51, 121,
       163,  73, 171, 161, 116, 131, 176, 137,  56,  96, 111,  58, 113,
        25,  83, 119, 105, 180, 106,  76,  92,  50, 134,  55,  68,  71,
        97,  95,  33,  57, 114, 166,  87,  79, 177,  10, 109,  23,  84,
       175,  34,  62,  24, 158, 110, 126, 135,  80,  16,  31,  48,  30,
       102,  14, 141, 104, 167,  67, 151, 136, 138,  22, 103, 139,  37,
         1, 181, 143, 115,  46,   2,  47,   9,  61, 155,  86, 152,  54,
       153, 147, 149, 144, 156, 154,   4, 165, 170, 112,  75,  17, 174,
        85], dtype=int64)
        '''

    ########
    T, N = 237, 235
    #x = np.random.normal(0, 1, size = (T, N))
    S, pnames = get_OL_tickers_close(T, N)
    np.argwhere(np.isnan(S))
    S[204, 109]=S[203, 109]

    cov0 = np.cov(S, rowvar=0, ddof=1)
    q = float(S.shape[0])/float(S.shape[1])#T/N
    #eMax0, var0 = mp.findMaxEval(np.diag(eVal0), q, bWidth=.01)

    corr0 = mp.cov2corr(cov0)
    eVal0, eVec0 = mp.getPCA(corr0)
    bWidth = best_bandwidth.findOptimalBWidth(np.diag(eVal0))
    
    min_var_markowitz = mc.optPort(cov1_d, mu1).flatten()
    min_var_NCO = pc.optPort_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()
    
    
    ##################
    # Test if bWidth found makes sense
    pdf0 = mp.mpPDF(1., q=T/float(N), pts=N)
    pdf1 = mp.fitKDE(np.diag(eVal0), bWidth=bWidth['bandwidth']) #empirical pdf
    #pdf1 = mp.fitKDE(np.diag(eVal0), bWidth=0.1)

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(np.diag(eVal0), density = True, bins=50) # Histogram the eigenvalues
    plt.plot(pdf0.keys(), pdf0, color='r', label="Marcenko-Pastur pdf")
    plt.plot(pdf1.keys(), pdf1, color='g', label="Empirical:KDE")
    plt.legend(loc="upper right")
    plt.show()
        
    N = 1000
    T = 10000
    x = np.random.normal(0, 1, size = (T, N))
    cor = np.corrcoef(x, rowvar=0)
    eVal0, eVec0 = mp.getPCA(cor)
    bWidth = best_bandwidth.findOptimalBWidth(np.diag(eVal0))
    #{'bandwidth': 4.328761281083057}
    ###############
    
    
    bWidth=0.1
    cov1_d = mc.deNoiseCov(cov0, q, bWidth)
    mu1 = None

    min_var_markowitz = mc.optPort(cov1_d, mu1).flatten()
    min_var_NCO = pc.optPort_nco(cov1_d, mu1, int(cov1_d.shape[0]/2)).flatten()
    