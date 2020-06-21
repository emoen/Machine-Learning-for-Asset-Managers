import yfinance as yf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


def test_charting(ticker, dir):
    pho = yf.Ticker(ticker)  # "pho.ol"
    # pho.info
    hist = pho.history(period="max")

    scaler = MinMaxScaler(feature_range=(0, 1))
    close = hist.Close.values
    close = close.reshape(-1, 1)
    open = hist.Open.values
    open = open.reshape(-1, 1)
    high = hist.High.values
    high = high.reshape(-1, 1)
    low = hist.Low.values
    low = low.reshape(-1, 1)
    volume = hist.Volume

    nans = np.argwhere(np.isnan(close))
    if len(nans) > 0:  # set previuous close as close when nan
        prev_close = close[(nans[0] - 1)]
        prev_open = open[(nans[0] - 1)]
        prev_high = high[(nans[0] - 1)]
        prev_low = low[(nans[0] - 1)]
        for i in range(0, len(nans)):
            close[nans[i]] = prev_close
            open[nans[i]] = prev_open
            high[nans[i]] = prev_high
            low[nans[i]] = prev_low
            volume[nans[i]] = 0

    norm_close = scaler.fit_transform(close)
    norm_open = scaler.fit_transform(open)
    norm_high = scaler.fit_transform(high)
    norm_low = scaler.fit_transform(low)

    pho_norm = pd.DataFrame(
        columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ewm26', 'ewm12', 'macd', 'signal'])
    pho_norm.time = hist.index
    pho_norm.open = norm_open
    pho_norm.close = norm_close
    pho_norm.high = norm_high
    pho_norm.low = norm_low
    pho_norm.volume = volume = hist.Volume
    pho_norm.ewm26 = pho_norm.close.ewm(span=26, adjust=False).mean()
    pho_norm.ewm12 = pho_norm.close.ewm(span=12, adjust=False).mean()
    pho_norm.macd = pho_norm.ewm12 - pho_norm.ewm26
    pho_norm.signal = pho_norm.macd.ewm(span=9, adjust=False).mean()

    pho_norm.volume = pho_norm.volume.fillna(0)

    #for i in range(0, (len(pho_norm) - 30)):
    #    chart_to_image(pho_norm[i:30 + i], dir + '/' + dir + str(i) + '.png')

    # chart_to_image(pho_norm.tail(30), 'pho/pho_tail.png')
    # print(pho_norm.shape)

    # arr = chart_to_arr(pho_norm.tail(30))
    # assert arr.shape == (3, 224, 224)


def ol_tickers():
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
        covariance_matrix = (np.dot(de_meaned_S.T, de_meaned_S))/(n*T)
        # The eigen-values of the covariance matrix is distributed like Marcenko-Pasture dist.
        #any any eigenvalues outside distribution is signal else noise.





if __name__ == '__main__':
    ol_tickers()