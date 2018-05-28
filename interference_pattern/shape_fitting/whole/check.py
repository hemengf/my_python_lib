import numpy as np
d = np.load('goodness.npy').item()
print d
print min(d, key=d.get)
