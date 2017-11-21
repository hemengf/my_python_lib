from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from boundaryv.brownian_gas import findnearest


class Particle:
	def __init__(self):
		self.position = np.array([0.,0.])
		self.velocity = np.array([0.,0.])
		self.repelforce = np.zeros(2)

	def accelerate(self, acceleration):
		self.velocity += acceleration
	def move(self,velocity):
		self.position += self.velocity

class Disk(Particle):
	def __init__(self, mass, radius):
		Particle.__init__(self) # __init__ of base class is overwritten by subclass __init__
		self.radius = radius
		self.mass = mass

def touch(particle1pos, particle2pos, particle1size, particle2size):
	""" Calculate overlap of 2 particles """ 
	overlap = -np.linalg.norm(particle1pos-particle2pos)+(particle1size + particle2size)
	if overlap > 0.:  
		return overlap
	else:
		return 0.
def tchbnd(particle_position, radius, boxsize):
	# boxsize is a tuple: horizontal, vertical
	tchbndlist = [0,0,0,0]  # [W,N,E,S]
	xtemp = particle_position[0]
	ytemp = particle_position[1]
	if xtemp<=radius:  
		tchbndlist[0] = 1
	if xtemp>=(boxsize[0]-radius):
		tchbndlist[2] = 1
	if ytemp>=(boxsize[1]-radius):
		tchbndlist[1] = 1
	if ytemp<=radius:
		tchbndlist[3] = 1
	return tchbndlist


class Environment:
	def __init__(self, boxsize, lower_doorbnd, upper_doorbnd, totnum, dt, repel_coeff, friction_coeff, belt_velocity):
		# boxsize is a tuple: horizontal, vertical
		# lower_doorbnd is a np array coordinate
		self.boxsize = boxsize
		self.lower_doorbnd = lower_doorbnd
		self.upper_doorbnd = upper_doorbnd
		self.totnum = totnum
		self.particle_position_array = np.empty((self.totnum,2))
		self.particle_position_array[:] = np.nan
		self.particle_list = [0]*self.totnum
		self.dt = dt
		self.repel_coeff = repel_coeff
		self.friction_coeff = friction_coeff
		self.belt_velocity = belt_velocity

	def create_disks(self, mass, radius):
		print 'Creating particles...'
		for n in range(0,self.totnum):
			overlap = 1
			out_of_bnd = 1
			while overlap or out_of_bnd:
				disk = Disk(mass, radius)
				disk.position[0] = np.random.uniform(radius, self.boxsize[0]-radius)
				disk.position[1] = np.random.uniform(radius, self.boxsize[1]-radius)
				try:
					nearest_idx = findnearest(disk.position, self.particle_position_array)
					overlap = touch(disk.position, self.particle_position_array[nearest_idx], radius, radius)
					tchbndlist = tchbnd(disk.position, disk.radius, self.boxsize)
					out_of_bnd = sum(tchbndlist)
				except ValueError:
					# just for the first particle creation, self.particle_position_array could be all nan, which would raise a ValueError when using findnearest
					break
			self.particle_position_array[n,:] =  disk.position	
			self.particle_list[n] = disk
			processbar.processbar(n+1, self.totnum, 1)
	def read_positions(self, mass, radius):
		self.particle_position_array = np.load('initial_positions_real_try.npy')
		for n in range(0, self.totnum):
			disk = Disk(mass, radius)
			disk.position = self.particle_position_array[n,:]
			self.particle_list[n] = disk
	def visualize(self): 
		fig = plt.figure(figsize=(8.0,5.0))
		for disk in self.particle_list:
			circle = plt.Circle(disk.position, disk.radius, fill = False, linewidth=0.3)
			fig.gca().add_artist(circle)
			plt.plot((0,0),(0,self.lower_doorbnd[1]), 'k', linewidth=0.3)
			plt.plot((0,0),(self.upper_doorbnd[1], self.boxsize[1]), 'k', linewidth=0.3)
		plt.axis([-0.3*self.boxsize[0],self.boxsize[0], 0,self.boxsize[1]])
		plt.axes().set_aspect('equal')
		#plt.show()
	def assign_repel(self):
		repel_list = []
		overlap_list = []
		overlapsum = 0.
		for particle in self.particle_list:
			particle.repelforce = np.zeros(2)
			# Clear assigned forces from the last iteration.
		for n, particle in enumerate(self.particle_list):
			for i, particle_position in enumerate(self.particle_position_array):
				if i != n:  # Exclude itself
					overlap = touch(particle.position, particle_position, particle.radius, particle.radius)
					unit_vector = (particle.position-particle_position)/np.linalg.norm((particle.position-particle_position))
					particle.repelforce += self.repel_coeff * unit_vector * overlap
					overlapsum += overlap
			repel_list.append(particle.repelforce[0])
			repel_list.append(particle.repelforce[1])
			overlap_list.append(overlapsum)
		return repel_list, overlap_list
	def assign_beltfriction(self):
		friction_list = []
		for n, particle in enumerate(self.particle_list):
			unit_vector = (self.belt_velocity-particle.velocity)/np.linalg.norm((self.belt_velocity-particle.velocity))
			particle.beltfriction = 9.8 * particle.mass * self.friction_coeff * unit_vector 
			friction_list.append(particle.beltfriction[0])
			friction_list.append(particle.beltfriction[1])
		return friction_list
	def wall_interact(self):
		for particle in self.particle_list:
			if particle.position[0]<=particle.radius and particle.position[1]<=self.upper_doorbnd[1] and particle.position[1]>=self.lower_doorbnd[1]:   #  takes care of the situation when a particle hits the corners of the doorbnd
				if np.linalg.norm(particle.position-self.lower_doorbnd) <= particle.radius and particle.position[1]>=self.lower_doorbnd[1]:
					unit_vector = -(particle.position-self.lower_doorbnd)/np.linalg.norm(particle.position-self.lower_doorbnd)
					normal_velocity = np.dot(particle.velocity,unit_vector)
					if normal_velocity > 0:
						particle.velocity = particle.velocity - unit_vector * normal_velocity
				if np.linalg.norm(particle.position-self.upper_doorbnd) <= particle.radius and particle.position[1]<=self.upper_doorbnd[1]:
					unit_vector = -(particle.position-self.upper_doorbnd)/np.linalg.norm(particle.position-self.upper_doorbnd)
					normal_velocity = np.dot(particle.velocity,unit_vector)
					if normal_velocity > 0:
						particle.velocity = particle.velocity - unit_vector * normal_velocity
			elif particle.position[0] > 0.:  #  takes care of the situation when a particle hits other part of the wall
				tchbndlist = tchbnd(particle.position, particle.radius, self.boxsize)
				if tchbndlist[0] * particle.velocity[0] < 0.:
					particle.velocity[0] = 0.
				if tchbndlist[2] * particle.velocity[0] > 0.:
					particle.velocity[0] = 0.
				if tchbndlist[1] * particle.velocity[1] > 0.:
					particle.velocity[1] = 0.
				if tchbndlist[3] * particle.velocity[1] < 0.:
					particle.velocity[1] = 0.
				
	def accelerate(self):
		for particle in self.particle_list:
			particle.force = particle.beltfriction + particle.repelforce
			particle.velocity += self.dt*particle.force/particle.mass
	def move(self):
		for n, particle in enumerate(self.particle_list):
			particle.position += self.dt*particle.velocity
			self.particle_position_array[n,:] = particle.position
	def update(self):
		repel_list, overlap_list = self.assign_repel()
		#f = open('./resultsfile.txt', 'a')
		#print >> f, ''.join('{:<+10.2f}'.format(e) for e in repel_list)

		friction_list = self.assign_beltfriction()
		#f = open('./resultsfile.txt', 'a')
		#print >> f, ''.join('{:<+10.2f}'.format(e) for e in friction_list)
		
		#result_list = overlap_list + repel_list+friction_list
		#f = open('./resultsfile.txt', 'a')
		#print >> f, ''.join('{:<+7.1f}'.format(e) for e in result_list)


		self.accelerate()
		self.wall_interact()
		self.move()

	def measure_pass(self):
		pass_number = sum(e<0 for e in self.particle_position_array[:,0])
		return pass_number

if  __name__ == '__main__':
	import matplotlib.pyplot as plt
	import processbar
	import os
	import subprocess
	import time
	start = time.time()
	open('resultsfile.txt', 'w').close()
	env = Environment(boxsize=(0.6,0.4), \
			lower_doorbnd=np.array([0,0]), \
			upper_doorbnd=np.array([0,0.06]), \
			totnum=500, \
			dt=0.005, \
			repel_coeff=100, \
			friction_coeff=0.5, \
			belt_velocity=np.array([-0.02,0]))
	#env.create_disks(mass = 0.005, radius = 0.010)
	env.read_positions(mass = 0.005, radius = 0.010)
	
	for disk in env.particle_list:
		print disk.position
	totframe = 1200 
	passnumber_list = []
	for i in range(totframe):
		env.update()
		if i%3==0:
			env.visualize()
			plt.savefig('./movie_try/'+'{:4.0f}'.format(i)+'.tif', dpi = 200)
			plt.close()
			pass_number = env.measure_pass()
			passnumber_list.append(pass_number)
		#if i == 2000:
		#	np.save('initial_positions_real_try', env.particle_position_array)

		processbar.processbar(i+1, totframe, 1)
	#subprocess.call('less resultsfile.txt', shell=False)
	g = open('passnumber.txt', 'w')
	print >> g, passnumber_list
	np.save('passnumber_list_real', passnumber_list)
	end = time.time()
	print end-start
	#plt.plot(passnumber_list)
	#plt.show()
