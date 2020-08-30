import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore

# code snippet 8.1 - experimental validation of the false strategy theorem
def get ExpectedMaxSR(nTrials, meanSR, stdSR):
    #Expected max SR, controlling for SBuMT
    emc = 0.477215664901532860606512090082402431042159336
    sr0 = (1-emc)*norm.ppf(1-1./nTrials)+emc*norm.ppf(1-(nTrials*np.e)**-1)
    sr0 = meanSR + stdSR*sr0
    return sr0

def getDistMaxSR(nSims, nTrials, stdSR, meanSR):
    #Monte carlo of max{SR} on nTrials, from nSims simulations
    rng = np.random.RandomState()
    out = pd.DataFrame()
    for nTrials_ in nTrials:
        # 1) simulated sharpe ratios
        sr = pd.DataFrame(rng.randn(nSims, nTrials_))
        sr = sr.sub(sr.mean(axis=1), axis=0) #center
        sr = sr.div(sr.std(axis=1), axis=0) #scale
        sr = meanSR+sr*stdSR
        #2) store output
        out_ = sr.max(axis=1).to_frame('max{SR}')
        out_['nTrials'} = nTrials_
        out = out.append(out_, ignore_index=True)
    return out

if __name__ == '__main__': 
    nTrials = list(set(np.logspace(1, 6, 1000).astype(int)))
    nTrials.sort()
    sr0 = pd.Series({i:getExpectedMaxSR(i, meanSR=0, stdSR=1) for i in nTrials})
    sr1 = getDistMaxSR(nSims=1E3, nTrials = nTrials, meanSR=0, stdSR=1)