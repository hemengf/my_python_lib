from __future__ import division
import numpy as np
import sys
def split_concatenate(img1, img2, angle, sp):
    """
    Takes two pictures of (e.g. red and green) interference patterns and 
    concatenate them in a split screen fashion for easy comparison.
    
    The split line is the line that passes sp===split_point and with an
    inclination of angle.
   """

    img1cp = np.copy(img1)
    img2cp = np.copy(img2)
    if img1cp.shape != img2cp.shape:
        print "I can't deal with pictures of difference sizes..."
        sys.exit(0)
    angle = angle*np.pi/180
    for j in range(img1cp.shape[1]):
        ic = -np.tan(angle)*(j-sp[0])+sp[1]
        for i in range(img1cp.shape[0]):
            if i>=ic:
                img1cp[i,j] = 0
            else:
                img2cp[i,j] = 0
    img = np.maximum(img1cp,img2cp)
    return img

if __name__ == "__main__":
    """
    img1 is above img2
    """
    import numpy as np
    import cv2
    img1 = cv2.imread('catreference.tif', 0) 
    img2 = cv2.imread('greenveo2_f358enhanced.tif',0)
    img = split_concatenate(img1,img2, angle =96.759,\
            sp=(674,175)) 
    cv2.imwrite('catreference.tif', img)
    print "Finished!"
