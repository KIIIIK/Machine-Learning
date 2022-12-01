import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

"""
random walk Metropolis-Hastings algorithm
产生标准正态分布随机数
"""
delta = 0.5
N = 1000
x = np.zeros(N)
x.shape
x[0] = 0

for i in range(1, N):
    eps = np.random.uniform(-delta, delta, size=1)
    y = x[i-1] + eps
    alpha = min(norm.logpdf(y) - norm.logpdf(x[i-1]), 0)
    u = np.random.uniform(0, 1, size=1)
    
    if np.log(u) <= alpha:
        x[i] = y
    else:
        x[i] = x[i-1]

plt.hist(x, edgecolor="black")
plt.show()





