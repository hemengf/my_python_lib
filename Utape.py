from __future__ import division
import numpy as np
import sys
dt = sys.argv[1] #0.005
while 1:    
    try:
        intv = input('intervels(pix): ')
        s = np.mean(intv)
        percenterr = np.std(intv)/s
        break
    except Exception as e:
        print e 
while 1:    
    try:
        R = input('mm/pix ratio: ')
        r = float(R[0])/float(R[1])
        U = s*r/float(dt)
        dU = percenterr*U
        break
    except Exception as e:
        print e
print '[average intv pix', s, 'pix]'
print 'U=', U,'mm/s'
print 'dU=', dU, 'mm/s'


