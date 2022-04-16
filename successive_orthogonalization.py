import numpy as np

def regress(x,z_list):
    return [np.dot(z,x)/np.dot(z,z) for z in z_list]

def gram_schmidt(x,y):
    n = len(x[0])
    p = len(x)
    x = np.vstack([np.ones(n),x])

    z = [np.ones(n)]*(p+1)

    for j in range(1,p+1):
        hat_gamma = regress(x[j],[z[k] for k in range(j)])
        z[j]=x[j] - sum([hat_gamma[k]*z[k] for k in range(j)])

    hat_beta = regress(y,[z[p]])
    return hat_beta[0]

if __name__ == "__main__":
    x = np.random.rand(6, 9)
    y = np.random.rand(9)
    hat_beta = gram_schmidt(x,y)
    print(hat_beta)