import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import bernoulli
from sympy import symbols, Eq, solve


fig, ax = plt.subplots(figsize=(6, 6))
plt.ylim([0, 1])


def proba(x, y):
    return ((x**(1/10))*(y))


x = np.arange(.001, 1, .001)
y = ((0.5)/(x**(1/10)))
plt.plot(x, y)

for i in range(1000):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    p = proba(x, y)
    z = bernoulli.rvs(p)

    if z == 0:
        ax.plot(x, y, 'bo', markersize=3)
    else:
        ax.plot(x, y, 'ro', markersize=3)


plt.show()
