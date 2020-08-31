import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore

# code snippet 8.1 - experimental validation of the false strategy theorem
def getExpectedMaxSR(nTrials, meanSR, stdSR):
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
        sr = pd.DataFrame(rng.randn(nSims, nTrials_)) #Return a sample (or samples) from the “standard normal” distribution.
        sr = sr.sub(sr.mean(axis=1), axis=0) #center
        sr = sr.div(sr.std(axis=1), axis=0) #scale
        sr = meanSR+sr*stdSR
        #2) store output
        out_ = sr.max(axis=1).to_frame('max{SR}')
        out_['nTrials'] = nTrials_
        out = out.append(out_, ignore_index=True)
    return out

if __name__ == '__main__': 
    nTrials = list(set(np.logspace(1, 6, 1000).astype(int)))
    nTrials.sort()
    sr0 = pd.Series({i:getExpectedMaxSR(i, meanSR=0, stdSR=1) for i in nTrials}) #prior
    sr1 = getDistMaxSR(nSims=1000, nTrials = nTrials, meanSR=0, stdSR=1) #observed
    
    #dashes = [10, 5, 100, 5] 
    fig, ax = plt.subplots()
    line1, = ax.plot(range(0, nTrials), sr0, '--', linewidth=2, label='E[max{SR}} (prioer)')
    #line1.set_dashes(dashes)

    #line2, = ax.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5],
    #                 label='Dashes set proactively')
    #line1, = ax.plot(range(0, nTrials), sr0, '--', linewidth=2, label='E[max{SR}} (prioer)')
    plt.contour(range(0, nTrials), sr1, 20, cmap='RdGy')
    plt.colorbar();

    ax.legend(loc='lower right')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('Figure 6.5 Clustered MDI')
    plt.show()