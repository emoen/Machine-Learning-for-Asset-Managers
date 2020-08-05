# -*- coding: utf-8 -*-
import statsmodels.api as sm1
import numpy as np

#Trend scanning method

#code snippet 5.1
def tValLinR(close):
    #tValue from a linear trend
    x = np.ones((close.shape[0],2))
    x[:,1] = np.arange(close.shape[0])
    ols = sm1.OLS(close, x).fit()
    return ols.tvalues[1]
    
#code snippet 5.2
def getBinsFromTrend(molecule, close, span):
    '''
    Derive labels from the sign of t-value of trend line
    output includes:
      - t1: End time for the identified trend
      - tVal: t-value associated with the estimated trend coefficient
      - bin: Sign of the trend
    '''
    out = pd.DataFrame(index=molecule, columns=['t1', 'tVal', 'bin'])
    for dt0 in molecule:
        df0 = pd.Series()
        iloc0 = close.index.get_loc(dt0)
        if iloc0+max(hrzns) > close.shape[0]:
            continue
        for hrzn in hrzns:
            dt1 = close.index[iloc0+hrzn-1]
            df1 = close.loc[dt0:dt1]
            df0.loc[dt1] = tValLinR(df1.values)
        dt1=df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, ['t1', 'tVal', 'bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1]) #prevent leakage
    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    return out.dropna(subset=['bin'])
    
#snippet 5.3
df0 = pd.Series(np.random.normal(0, .1, 100)).cumsum()
df0 += np.sin(np.linspace(0, 10, df0.shape[0]))
mp1.scatter(df1.index, df0.loc[df1.index].values, mp1.savefig('fig5.1.png)); mp1.clf(); mp1.close()
mp1.scatter(df1.index, df0.loc[df1.index].values, c=c, cmap='viridis')
    