import matplotlib.pyplot as plt
import processbar
import os
import subprocess
import time
from door_position.disks import *

for batchiter in range(8):
	print 'processing iteration {:d}'.format(batchiter)
	start = time.time()
	env = Environment(boxsize=(0.6,0.4), \
			lower_doorbnd=np.array([0,batchiter*0.02+0.01]), \
			upper_doorbnd=np.array([0,batchiter*0.02+0.06+0.01]), \
			totnum=500, \
			dt=0.005, \
			repel_coeff=100, \
			friction_coeff=0.5, \
			belt_velocity=np.array([-0.05,0]))
	#env.create_disks(mass = 10, radius = 5)
	env.read_positions(mass = 0.005, radius = 0.010)

	#for disk in env.particle_list:
	#	print disk.position
	totframe = 1200 
	passnumber_list = []
	if not os.path.exists('./passnumber_door_position_v5cm'):
		os.makedirs('./passnumber_door_position_v5cm')
	for i in range(totframe):
		env.update()
		if i%3==0:
			#env.visualize()
			#plt.savefig('./movie32/'+'{:4.0f}'.format(i)+'.tif', dpi = 300)
			#plt.close()
			pass_number = env.measure_pass()
			passnumber_list.append(pass_number)
		#if i == 1000:
		#	np.save('initial_positions', env.particle_position_array)

		processbar.processbar(i+1, totframe, 1)
	#subprocess.call('less resultsfile.txt', shell=False)
	#g = open('passnumber.txt', 'w')
	#print >> g, passnumber_list
	np.save('./passnumber_door_position_v5cm/passnumber_list_append {:d}'.format(batchiter), passnumber_list)
	end = time.time()
	print 'time consumption', end-start,'s'
	#plt.plot(passnumber_list)
	#plt.show()
