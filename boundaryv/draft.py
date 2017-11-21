import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import random
boxsize = 1000
class Particle:
	def __init__(self, particle_pos, size):
		self.x = particle_pos[0]
		self.y = particle_pos[1] 
		self.orientation = random.uniform(0,2*np.pi) 
		self.size = size

def touch(particle1pos, particle2pos, particle1size, particle2size):
    if np.linalg.norm(particle1pos-particle2pos) <= particle1size + particle2size:
		return 1
    else:
		return 0

def findnearest(particle, particle_array):
    dist_array = np.sum((particle - particle_array)**2, axis=1)
    return np.nanargmin(dist_array)
    
def create_multi_particles(totnum):
	boxsize = 1000
	particle_array = np.empty((totnum,2))
	particle_array[:] = np.NAN
	particlesize = 10
	x= random.uniform(particlesize, boxsize-particlesize)
	y = random.uniform(particlesize, boxsize-particlesize)
	particle_array[0,:] = np.asarray((x,y))
	for n in range(1,totnum):
		touchflg = 1
		particlesize = 10
		failcount = -1
		while touchflg == 1:
			failcount+=1
			x = random.uniform(particlesize, boxsize-particlesize)
			y = random.uniform(particlesize, boxsize-particlesize)
			particle = np.asarray((x,y))
			nearest_idx = findnearest(particle,particle_array)
			touchflg = touch(particle_array[nearest_idx], particle, particlesize, particlesize)
		particle_array[n,:] = np.asarray((x,y))
	return particle_array, failcount

if __name__ == '__main__':
	totnum = 100
	particle_array, failcount = create_multi_particles(totnum)
	fig = plt.figure()
	for n in range(totnum):
		circle = plt.Circle((particle_array[n,0], particle_array[n,1]), 10, fill=False)
		fig.gca().add_artist(circle)
	plt.axis([0,1000,0,1000])
	plt.axes().set_aspect('equal')
	plt.show()
	print failcount

