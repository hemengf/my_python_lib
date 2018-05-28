from boundaryv.brownian_gas import touch, findnearest
from door_position.disks import tchbnd
import numpy as np
import matplotlib.pyplot as plt
import os
import progressbar

class Elephant_foot():
    def __init__(self, radius, velocity):
        self.position = np.array([0.,0.])
        self.radius = radius
        self.velocity = velocity 
    def expand(self, dt):
        self.radius += self.velocity * dt
class Environment():
    def __init__(self, boxsize, totnum, dt, initial_radius, velocity):
        self.boxsize = boxsize
        self.totnum = totnum
        self.foot_list = [0] * self.totnum 
        self.foot_position_array = np.empty((self.totnum,2)) 
        self.foot_position_array[:] = np.nan
        self.dt = dt
        self.initial_radius = initial_radius
        self.velocity = velocity
		
    def create_feet(self):
        print 'Creating elephant feet...'
        if os.path.exists('./initial_particles.npy') & os.path.exists('./initial_positions.npy'):
            print 'Reading saved initial conditions...'
            self.foot_list = np.load('initial_particles.npy')
            self.foot_position_array = np.load('initial_positions.npy')
        else:
            for n in range(0,self.totnum):
                out_of_bnd = 1
                overlap = 1
                while out_of_bnd or overlap:
                    foot = Elephant_foot(self.initial_radius, self.velocity) 
                    foot.position[0] = np.random.uniform(foot.radius, self.boxsize[0]-foot.radius) 
                    foot.position[1] = np.random.uniform(foot.radius, self.boxsize[1]-foot.radius) 
                    try:
                        nearest_idx = findnearest(foot.position, self.foot_position_array)
                        nearest_foot = self.foot_list[nearest_idx]
                        overlap = touch(foot.position, self.foot_position_array[nearest_idx],foot.radius,nearest_foot.radius)
                        tchbndlist = tchbnd(foot.position, foot.radius, self.boxsize)
                        out_of_bnd = sum(tchbndlist)
                    except ValueError:
                        break
                self.foot_list[n] = foot 
                self.foot_position_array[n,:] = foot.position
                progressbar.progressbar_tty(n+1, self.totnum, 1)
            np.save('initial_particles',self.foot_list)
            np.save('initial_positions',self.foot_position_array)
    def visualize(self): 
        fig = plt.figure(figsize=(8.0,5.0))
        for foot in self.foot_list:
            circle = plt.Circle(foot.position, foot.radius, fill = True, linewidth=0.3)
            fig.gca().add_artist(circle)
        plt.axis([0,self.boxsize[0], 0,self.boxsize[1]])
        plt.axes().set_aspect('equal')
        plt.savefig('./movie/'+'{:4.0f}'.format(i)+'.tif', dpi = 300)
    def expand(self):
        for n, footn in enumerate(self.foot_list):
            overlap = 0
            for i , footi in enumerate(self.foot_list):
                if n != i: 
                    overlap += touch(footn.position, footi.position,footn.radius,footi.radius)
            tchbndlist = tchbnd(footn.position, footn.radius, self.boxsize)
            out_of_bnd = sum(tchbndlist)
            #if overlap + out_of_bnd == 0:
            if 1:
                footn.radius += self.velocity * self.dt
    def update(self):
        self.expand()
if  __name__ == "__main__":
    import matplotlib.pyplot as plt
    import progressbar 
    import os
    import subprocess
    import time
    import os
    if not os.path.exists('./movie/'):
        os.makedirs('./movie/')
    start = time.time()
    env = Environment(boxsize=(30,30), \
                    totnum=200, \
                    dt=0.03, \
                    initial_radius=0.1, \
                    velocity=0.5)
    env.create_feet()
    #env.read_positions(mass = 10, radius = 5)
    array = []

    totframe = 200 
    for i in range(totframe):
        env.update()
        if i%3==0:
            env.visualize()
            plt.close()
        #if i == 1000:
        #    np.save('initial_positions', env.particle_position_array)

        progressbar.progressbar_tty(i+1, totframe, 1)
    #subprocess.call('less resultsfilekjk.txt', shell=False)
    for foot in env.foot_list:
        #print foot.position
        array.append(foot.radius)
    plt.hist(array,13)
    plt.show()
    end = time.time()
    print end-start
	
		
