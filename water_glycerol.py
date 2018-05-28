from __future__ import division
import numpy as np
from scipy.optimize import fsolve

def mu(Cm,T):
    a = 0.705-0.0017*T
    b = (4.9+0.036*T)*np.power(a,2.5)
    alpha = 1-Cm+(a*b*Cm*(1-Cm))/(a*Cm+b*(1-Cm))
    mu_water = 1.790*np.exp((-1230-T)*T/(36100+360*T))
    mu_gly = 12100*np.exp((-1233+T)*T/(9900+70*T))
    return  np.power(mu_water,alpha)*np.power(mu_gly,1-alpha)

def glycerol_mass(T,target_viscosity=200):
    def mu_sub(Cm,T):
        return mu(Cm,T)-target_viscosity
    x = fsolve(mu_sub,1,args=(T),xtol=1e-12)
    return x

Temperature = 22.5
Target_viscosity = 50 


print 'glycerol mass fraction %0.3f%%'%(glycerol_mass(Temperature,Target_viscosity)[0]*100)


