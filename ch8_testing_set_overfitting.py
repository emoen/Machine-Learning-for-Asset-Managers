import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore

# code snippet 8.1 - experimental validation of the false strategy theorem
def get ExpectedMaxSR(nTrials, meanSR, stdSR):
    #Expected max SR, controlling for SBuMT
    emc = 0.477215664901532860606512090082402431042159336
    sr0 = (1-emc)*norm.ppf(1-1./nTrials)+emc*norm.ppf(1-(nTrials*np.e)**-1)