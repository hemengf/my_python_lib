from __future__ import division
import sys
contrast='uncalculated'
if len(sys.argv)>1:
    contrast = (float(sys.argv[1])-float(sys.argv[2]))/(float(sys.argv[1])+float(sys.argv[2]))
print contrast


