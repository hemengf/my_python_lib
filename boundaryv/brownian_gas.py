from __future__ import division
import progressbar
import matplotlib.pyplot as plt
import numpy as np
class gas:
	def __init__(self):
		pass
class Dimer(gas):
	def __init__(self, mass, radius, restlength):
		self.position1 = np.zeros(2)
		self.position2 = np.zeros(2) 
		self.positionCOM = (self.position1 + self.position2)/2.0
		self.restlength = restlength
		self.length = restlength
		self.orientation = 0.
		self.force1 = np.array((0.,0.))
		self.force2 = np.array((0.,0.))
		self.velocity1= np.array((0.,0.))
		self.velocity2= np.array((0.,0.))
		self.kickforce1= np.array((0.,0.))
		self.kickforce2= np.array((0.,0.))
		self.repelforce1= np.array((0.,0.))
		self.repelforce2= np.array((0.,0.))
		self.bondforce1= np.array((0.,0.))
		self.bondforce2= np.array((0.,0.))
		self.dissipation1= np.array((0.,0.))
		self.dissipation2= np.array((0.,0.))
		self.radius = radius
		self.mass = mass
		
		
	def interact():
		pass
	def accelerate(self, acceleration1, acceleration2, anglechange):
		self.velocity1 += acceleration1
		self.velocity2 += acceleration2
	def move(self, velocity1, velocity2):
		self.position1 += self.velocity1
		self.position2 += self.velocity2
		
def touch(particle1pos, particle2pos, particle1size, particle2size):
	""" Calculate overlap of 2 particles """ 
	overlap = -np.linalg.norm(particle1pos-particle2pos)+(particle1size + particle2size)
	if overlap > 0.:  
		return overlap
	else:
		return 0.
def touchbnd(particle_position, radius, box_size):
	""" Tells if a particle touches the boundary """
	tchbndlist = [0,0,0,0]  # [W,N,E,S]
	xtemp = particle_position[0]
	ytemp = particle_position[1]
	if xtemp<=radius:  
		tchbndlist[0] = 1
	if xtemp>=(box_size-radius):
	#if xtemp>=8*radius:
		tchbndlist[2] = 1
	if ytemp>=(box_size-radius):
		tchbndlist[1] = 1
	if ytemp<=radius:
		tchbndlist[3] = 1
	return tchbndlist

def findnearest(particle, particle_array):
	""" Returns the nearest particle index """ 
	dist_array = np.sum((particle - particle_array)**2, axis=1)
	return np.nanargmin(dist_array)

class Environment:
	def __init__(self, boxsize, totnum, dt): 
		self.boxsize = boxsize
		self.totnum = totnum
		self.particle_position_array = np.empty((2*self.totnum,2))
		self.particle_position_array[:] = np.nan
		self.dimer_list = [0]*self.totnum
		self.orientationlist = [0]*self.totnum
		self.bondlist = [[(0.,0.),(0.,0.)]]*totnum
		self.removallist = []
		self.dt = dt
	def create_dimers(self, mass, radius, restlength):
		# Place the first dimer
		dimer = Dimer(mass, radius, restlength)
		dimer.position1 = np.random.uniform(radius, self.boxsize-radius, 2)
		#dimer.position1 = np.random.uniform(radius, 8*radius, 2)
		out_of_bnd = 1
		while out_of_bnd:
			dimer.orientation = np.random.uniform(0, 2*np.pi)
			xtemp = dimer.position1[0] + dimer.length*np.cos(dimer.orientation)
			ytemp = dimer.position1[1] + dimer.length*np.sin(dimer.orientation)
			#  Unless sum of tchbndlist is zero, particle is out of bnd
			out_of_bnd = sum(touchbnd((xtemp, ytemp), radius, self.boxsize))
		dimer.position2[0] = xtemp
		dimer.position2[1] = ytemp
		self.orientationlist[0] = dimer.orientation
		self.dimer_list[0] = dimer
		self.particle_position_array[0,:] = dimer.position1
		self.particle_position_array[1,:] = dimer.position2
		self.bondlist[0] = (dimer.position1[0],dimer.position2[0]),(dimer.position1[1],dimer.position2[1])
		
		# Create 2nd-nth dimmer without overlapping
		for n in range(1,self.totnum):
			overlap = 1
			# Create particle1
			failcount1 = 0
			while overlap:
				failcount1 += 1 
				dimer = Dimer(mass, radius, restlength)
				dimer.position1 = np.random.uniform(radius+1, self.boxsize-radius-1, 2)
				nearest_idx = findnearest(dimer.position1, self.particle_position_array)
				overlap = touch(dimer.position1, self.particle_position_array[nearest_idx], radius, radius)
				if failcount1 >= 100000:
					self.removallist.append(n)
					break
			# Create particle2
			out_of_bnd = 1
			overlap = 1
			failcount2 = 0
			while out_of_bnd or overlap:
				failcount2 += 1
				dimer.orientation = np.random.uniform(0, 2*np.pi)
				xtemp = dimer.position1[0] + dimer.length*np.cos(dimer.orientation)
				ytemp = dimer.position1[1] + dimer.length*np.sin(dimer.orientation)
				out_of_bnd = sum(touchbnd((xtemp, ytemp), radius, self.boxsize))
				nearest_idx = findnearest((xtemp, ytemp), self.particle_position_array)
				overlap = touch((xtemp, ytemp), self.particle_position_array[nearest_idx], radius, radius)
				if failcount2 >= 100000:
					self.removallist.append(n)  
					break
			dimer.position2[0] = xtemp
			dimer.position2[1] = ytemp
			self.particle_position_array[2*n,:] = dimer.position1
			self.particle_position_array[2*n+1, :] = dimer.position2
			self.dimer_list[n] = dimer
			self.orientationlist[n] = dimer.orientation
			self.bondlist[n] = (dimer.position1[0],dimer.position2[0]),(dimer.position1[1],dimer.position2[1])
			progressbar.progressbar_tty(n+1,self.totnum,1)
		# Update dimer_list and everything related for removal
		self.removallist = list(set(self.removallist))
		print 'updating dimerlist, removing', self.removallist, len(self.removallist), ''
		self.dimer_list = [i for j, i in enumerate(self.dimer_list) if j not in self.removallist]
		newlength = len(self.dimer_list)
		self.orientationlist = [0]*newlength
		self.bondlist = [[(0.,0.),(0.,0.)]]*newlength
		self.particle_position_array = np.empty((2*newlength,2))
		self.particle_position_array[:] = np.nan
		for n, dimer in enumerate(self.dimer_list):
			self.particle_position_array[2*n,:] = dimer.position1
			self.particle_position_array[2*n+1, :] = dimer.position2
			self.orientationlist[n] = dimer.orientation  # Given randomly upon creation
			self.bondlist[n] = (dimer.position1[0],dimer.position2[0]),(dimer.position1[1],dimer.position2[1])
		print 'now length of dimerlist', len(self.dimer_list)
	def visualize(self):
		fig = plt.figure()
		radius = self.dimer_list[0].radius
		for dimer in self.dimer_list:
			circle = plt.Circle(dimer.position1, radius, fill=False)
			fig.gca().add_artist(circle)
			circle = plt.Circle(dimer.position2, radius, fill=False)
			fig.gca().add_artist(circle)
		count = 0 
		for n, dimer in enumerate(self.dimer_list):
			plt.plot(self.bondlist[n][0],self.bondlist[n][1],'k')
			count += 1
		plt.axis([0, self.boxsize, 0, self.boxsize])
		plt.axes().set_aspect('equal')
		return count
	def kick(self,kickf):
		for n, dimer in enumerate(self.dimer_list):
			kickangle = self.orientationlist[n]
			dimer.kickforce1 = kickf*np.cos(kickangle), kickf*np.sin(kickangle)
			dimer.kickforce1 = np.asarray(dimer.kickforce1)
			dimer.kickforce2 = dimer.kickforce1
	def dissipate(self, coefficient):
		for n, dimer in enumerate(self.dimer_list,coefficient):
			dimer.disspation1 = -coefficient*dimer.velocity1
			dimer.disspation2 = -coefficient*dimer.velocity2
	def collide(self,repel_coefficient):
		for n, dimer in enumerate(self.dimer_list):
			radius = dimer.radius
			dimer.repelforce1 = np.zeros(2)
			dimer.repelforce2 = np.zeros(2)
			for i, particle_position in enumerate(self.particle_position_array):
				if i != 2*n:  # for particle1, make sure to exclude itself
					overlap1 = touch(dimer.position1, particle_position, radius, radius)
					unit_vector = (dimer.position1-particle_position)/np.linalg.norm((dimer.position1-particle_position))
					dimer.repelforce1 += repel_coefficient*unit_vector*overlap1
				if i != 2*n+1:  # for particle2, exclude itself
					overlap2 = touch(dimer.position2, particle_position, radius, radius)
					unit_vector = (dimer.position2-particle_position)/np.linalg.norm((dimer.position2-particle_position))
					dimer.repelforce2 += repel_coefficient*unit_vector*overlap2
	def bounce(self):
		radius = self.dimer_list[0].radius
		for dimer in self.dimer_list:
			tchbndlist = touchbnd(dimer.position1, radius, self.boxsize)
			if tchbndlist[0] * dimer.velocity1[0] < 0:
				dimer.velocity1[0] = 0.
			if tchbndlist[2] * dimer.velocity1[0] > 0:
				dimer.velocity1[0] = 0.
			if tchbndlist[1] * dimer.velocity1[1] > 0:
				dimer.velocity1[1] = 0.
			if tchbndlist[3] * dimer.velocity1[1] < 0:
				dimer.velocity1[1] = 0.
			tchbndlist = touchbnd(dimer.position2, radius, self.boxsize)
			if tchbndlist[0] * dimer.velocity2[0] < 0:
				dimer.velocity2[0] = 0.
			if tchbndlist[2] * dimer.velocity2[0] > 0:
				dimer.velocity2[0] = 0.
			if tchbndlist[1] * dimer.velocity2[1] > 0:
				dimer.velocity2[1] = 0.
			if tchbndlist[3] * dimer.velocity2[1] < 0:
				dimer.velocity2[1] = 0.
	def bond_deform(self,coefficient):
		for n, dimer in enumerate(self.dimer_list):
			bondlength = np.linalg.norm(dimer.position2-dimer.position1)
			deform = bondlength - dimer.restlength
			unit_vector = np.asarray((np.cos(self.orientationlist[n]), np.sin(self.orientationlist[n])))
			dimer.bondforce1 = coefficient*unit_vector*deform
			dimer.bondforce2 = -coefficient*unit_vector*deform 
	def accelerate(self):
		for dimer in self.dimer_list:
			dimer.force1 = dimer.kickforce1 + dimer.dissipation1 + dimer.bondforce1 + dimer.repelforce1
			dimer.velocity1 += self.dt*dimer.force1/dimer.mass
			dimer.force2 = dimer.kickforce2 + dimer.dissipation2 + dimer.bondforce2 + dimer.repelforce2
			dimer.velocity2 += self.dt*dimer.force2/dimer.mass
	def move(self):
		for dimer in self.dimer_list:
			dimer.position1 += self.dt*dimer.velocity1
			dimer.position2 += self.dt*dimer.velocity2
	def update(self,kickf,collide_coeff,dissipate_coeff,bond_coeff):
		self.kick(kickf)
		self.collide(collide_coeff)
		self.bond_deform(bond_coeff)
		self.dissipate(dissipate_coeff)
		self.accelerate()
		self.bounce()
		self.move()
		for n, dimer in enumerate(self.dimer_list):
			self.particle_position_array[2*n,:] = dimer.position1
			self.particle_position_array[2*n+1, :] = dimer.position2
			bond = dimer.position2-dimer.position1
			dimer.orientation = np.angle(bond[0]+1j*bond[1])
			self.orientationlist[n] = dimer.orientation
			self.bondlist[n] = (dimer.position1[0],dimer.position2[0]),(dimer.position1[1],dimer.position2[1])

		
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import progressbar
	env = Environment(500,totnum = 110, dt = 0.02)
	env.create_dimers(mass=10., radius=10., restlength=30.)
	print env.removallist
	print len(env.orientationlist)
	totframe = 30000
	for i in range(totframe):
		env.update(kickf=1,collide_coeff=10,dissipate_coeff=1,bond_coeff=10)
		if i%30 == 0 and i>3000:
			env.visualize()
			plt.savefig('./movie5/'+'{:4.0f}'.format(i/10)+'.tif')
			plt.close()
		progressbar.progressbar_tty(i+1,totframe,1)
