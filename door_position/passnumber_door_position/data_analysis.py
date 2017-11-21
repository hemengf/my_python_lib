import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
	iternum = 16 
	p = [0]*iternum
	fig, ax = plt.subplots()
	for i in range(iternum):
		s = np.load('passnumber_list {:d}.npy'.format(i))
		#ax.plot(range(len(s)), s)
		pp = np.polyfit(range(len(s)),s, 1)
		p[i] = pp[0]
	plt.plot(range(iternum), p)
	plt.show()
	
	
