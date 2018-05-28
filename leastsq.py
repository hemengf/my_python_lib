from __future__ import division
from scipy import stats
import numpy as np

def leastsq_unweighted(x,y):
    """
    y = A + Bx
    all inputs are np arrays
    """
    N = len(x)
    delta_unweighted = N*((x**2).sum())-(x.sum())**2
    A_unweighted = ((x*x).sum()*(y.sum())-x.sum()*((x*y).sum()))/delta_unweighted
    B_unweighted = (N*((x*y).sum())-(x.sum())*(y.sum()))/delta_unweighted
    sigmay_unweighted = np.sqrt((1/(N-2))*np.square(y-A_unweighted-B_unweighted*x).sum())
    sigmaA = sigmay_unweighted*np.sqrt((x**2).sum()/delta_unweighted)
    sigmaB = sigmay_unweighted*np.sqrt(N/delta_unweighted)
    return A_unweighted, B_unweighted,sigmaA,sigmaB,sigmay_unweighted

def leastsq_weighted(x,y,sigmax_exp, sigmay_exp):
    _,B_unweighted,_,_,sigmay_unweighted = leastsq_unweighted(x,y)
    sigmay_max = np.array([max(s,t) for (s,t) in zip(sigmay_unweighted*y/y,sigmay_exp)])
    sigmay_eff = np.sqrt((sigmay_max)**2+np.square(B_unweighted*sigmax_exp)) # use sigmay_unweighted or sigmay_exp of sigmay_max????
    w = 1/np.square(sigmay_eff)
    delta_weighted = w.sum()*((w*x*x).sum()) - np.square((w*x).sum())
    A_weighted = ((w*x*x).sum()*((w*y).sum())-(w*x).sum()*((w*x*y).sum()))/delta_weighted
    B_weighted = (w.sum()*((w*x*y).sum()) - (w*x).sum()*((w*y).sum()))/delta_weighted
    sigmaA_weighted = np.sqrt((w*x*x).sum()/delta_weighted)
    sigmaB_weighted = np.sqrt(w.sum()/delta_weighted)
    return A_weighted, B_weighted, sigmaA_weighted, sigmaB_weighted

def leastsq_unweighted_thru0(x,y):
    """ y = Bx """
    N = len(y)
    numerator = (x*y).sum()
    denominator = (x**2).sum()
    B_unweighted = numerator/denominator
    sigmay_unweighted = np.sqrt(((y-B_unweighted*x)**2).sum()/(N-1))
    sigmaB = sigmay_unweighted/np.sqrt((x**2).sum())
    return B_unweighted, sigmaB, sigmay_unweighted

def leastsq_weighted_thru0(x,y,sigmax_exp,sigmay_exp):
    B_unweighted,_,sigmay_unweighted = leastsq_unweighted_thru0(x,y)
    sigmay_max = np.array([max(s,t) for (s,t) in zip(sigmay_unweighted*y/y,sigmay_exp)])
    sigmay_eff = np.sqrt((sigmay_max)**2+np.square(B_unweighted*sigmax_exp)) # use sigmay_unweighted or sigmay_exp of sigmay_max????
    w = 1/np.square(sigmay_eff)
    numerator = (w*x*y).sum()
    denominator = (w*x*x).sum()
    B_weighted = numerator/denominator
    sigmaB_weighted = 1/np.sqrt((w*x*x).sum())
    return B_weighted, sigmaB_weighted

def chi2test(x,y,sigmax_exp,sigmay_exp):
    _,_,_,_,sigmay_unweighted = leastsq_unweighted(x,y)
    A_weighted,B_weighted,_,_ = leastsq_weighted(x,y,sigmax_exp,sigmay_exp)
    chi2 = (np.square((y-A_weighted-B_weighted*x)/(sigmay_exp))).sum()#has to use sigmay_exp, a reasonable estimate of exp error is crucial
    N = len(x)
    c = 2 # sigmay_unweighted is calculated from data;1 constraint
    reduced_chi2 = chi2/(N-c)
    prob = (1-stats.chi2.cdf(chi2,(N-c)))
    return reduced_chi2 


    
