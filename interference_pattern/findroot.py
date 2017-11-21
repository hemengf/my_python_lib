import scipy.optimize
def F(x):
    return x[0], x[1]
def g(x):
    return x-1

if __name__ == "__main__":
    import numpy as np
    sol = scipy.optimize.fsolve(F, np.array([1,1]))
    x0 = scipy.optimize.root(g, 0)
    print sol 
    print x0.x[0]
