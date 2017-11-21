from __future__ import division
import scipy.misc
import numpy as np
def partial_derivative_wrapper(func, var, point):
    """
    Returns the partial derivative of a function 'func' with
    respect to 'var'-th variable at point 'point'
    Scipy hasn't provided a partial derivative function.
    This is a simple wrapper from http://stackoverflow.com/questions/20708038/scipy-misc-derivative-for-mutiple-argument-function
    
    func: callable name
    var, point: the variable with respect to which and 
        the point at which partial derivative is needed.

    usage:
       df(x,y)/dx|(3,2)
       partial_derivative(f, 0, [3,2])

    CONFUSION: 'point' has to be a list. Using numpy array
    doesn't work.

    """
    
    args = point[:]
    def reduce_variable(x):
        """
        Returns a function where all except the 'var'-th variable 
        take the value of 'args'.

        """
        args[var] = x
        return func(*args)
    return scipy.misc.derivative(reduce_variable, point[var], dx=1e-6)

def derivative(f, x, dx=1e-6):
    return (f(x+dx)-f(x))/dx
    

def partial_derivative(f, x, y, dx=1e-6, dy=1e-6):
    """
    Usage: 

    for N points simultaneously: 
    partial_derivative(f, *'Nx2 array of points'.T)
    returns=np.array ([[df/dx1,df/dy1],
                       [df/dx2,df/dy2],
                       [df/dx3,df/dy3]
                             .
                             .
                             .
                       [df/dxN,df/dyN]])

    for 1 point:
    partial_derivative(f, *np.array([3,2]))
    returns np.array([df/dx,df/dy])
    """

    dfdx = (f(x+dx,y)-f(x,y))/dx
    dfdy = (f(x,y+dy)-f(x,y))/dy
    #try:
    #    result = np.empty((len(x),2))
    #    result[:,0] = dfdx
    #    result[:,1] = dfdy
    #except TypeError:
    #    result = np.empty((2,))
    #    result[0] = dfdx
    #    result[1] = dfdy
    
    result = np.array((dfdx, dfdy))
    return result.T

if __name__ == "__main__":
    import time
    import numpy as np
    def g(x):
        return x**2
    def f(x,y):
        return x**2 + y**3
    # df/dx should be 2x
    # df/dy should be 3y^2
    start = time.time()
    result = partial_derivative(f,*np.array([[3,1], [3,1],[3,2],[1,2],[0,2]]).T)
    result2 = partial_derivative(f, *np.array([3,1]))
    result3 = derivative(g,np.array([1,2,3]))
    print time.time()-start
    print "vectorized:", result
    print "single argument:", result2, type(result2)
