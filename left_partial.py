from __future__ import division
def derivative(f, x, dx=1e-2):
    return (f(x+dx)-f(x-dx))/(2*dx)

if __name__ == "__main__":
    from mpmath import *
    mp.dps =2 
    def f(x):
        return x**4
    print derivative(f, 1, dx=1e-8)-4
    print derivative(f, 1, dx=-1e-8)-4
    print diff(f,1.)

