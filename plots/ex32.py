import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

N = 30
p = 3

low = -2
high = 2

def powers(x):
    return np.hstack([x**i for i in range(p + 1)])


x = np.random.uniform(low=low, high=high, size=(N, 1))

design = powers(x)

beta = np.random.uniform(low=-1, high=1, size=(p + 1, 1))

variance = 0.2
epsilon = np.random.normal(0, variance, size=(N, 1))

response = np.dot(design, beta) + epsilon

model = sm.OLS(response, design)
results = model.fit()

betahat = results.params
betaconf = results.conf_int(alpha=0.05, cols=None)
covariance = results.cov_params()


def C_ab_1(x):
    mean = np.dot(x, betahat)
    sd = np.sqrt(np.dot(x, np.dot(covariance, x)))
    return [mean - 1.96 * sd, mean + 1.96 * sd]


def C_ab_2(x):
    return np.dot(x, betaconf)

x = np.arange(low, high, 0.0005)

x1 = np.vstack([powers(i) for i in x])

actual = np.hstack([np.dot(xi, beta) for xi in x1])
c_ab_1 = np.vstack([C_ab_1(xi) for xi in x1])
c_ab_2 = np.vstack([C_ab_2(xi) for xi in x1])

plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, actual, color="r", label="actual")
plt.plot(x, c_ab_1, color="b", label="c_a", linewidth=0.5)
plt.plot(x, c_ab_2, color="g", label="c_b", linewidth=0.5)

plt.legend()
plt.show()
