"""Black-Sholes with Europian Call-option"""


import numpy as np
from scipy.stats import norm
X0 = 100
K = 100
t = 0
r = 0.01
S = X0
E = K
sigma = 0.4
T = 1


denom = np.log(S/E) + (r + ((sigma**2)/2)*(T-t))
numer = sigma * np.sqrt(T-t)
d1 = denom / numer
d2 = d1 - (sigma * np.sqrt(T-t))
Nd1 = norm.cdf(d1)
Nd2 = norm.cdf(d2)
print(d1)
print(d2)
print(Nd1)
print(Nd2)

ee = E * np.exp(-r*(T-t))
CSt = (S * Nd1) - (ee * Nd2)
print(ee)
print(CSt)