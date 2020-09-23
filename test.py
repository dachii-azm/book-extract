import numpy as np

img1 = np.array([0, 0, 255])
img2 = np.array([3, 4, 5])
img3 = np.array([0, 5, 255])

test = np.array([0, 0, 255])

#a_compare = img1 == test

if(np.any(img3 != test)):
#if(img1 == test):
    print(True)
else:
    print(False)
