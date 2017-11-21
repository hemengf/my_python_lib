from skimage import morphology
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np

#label original image, im=uint8(0 and 255), labeled=uint8
im = plt.imread('../../Downloads/image.tif')
labeled, nr_objects = mh.label(im,np.ones((3,3),bool))
print nr_objects

#an example of removing holes. Should use labeled image 
im_clean = morphology.remove_small_objects(labeled)
labeled_clean, nr_objects_clean = mh.label(im_clean,np.ones((3,3),bool))
print nr_objects_clean
