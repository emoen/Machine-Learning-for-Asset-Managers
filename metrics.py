import numpy as np
scipy.stats as ss
from sklearn.metrics import mutual_info_score

cXY = np.histogram2d(x, y, bins)[0]
hX = ss.entropy(np.histogram(x, bins)[0]) #marginal 
hY = ss.entropy(np.histogram(y, bins)[0]) #marginal
iXY = mutual_info_score(None, None, contingency=cXY)
iXYn = iXY/min(hX, hY) #normalized mutual information
hXY = hX+hY - iXY #joint
hX_Y = hXY-hY #conditional
hY_X = hXY-hX #contitional
