import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


"""
Q3.4
"""
def matchPics(I1, I2, ratio=0.67):
    sigma = 0.139

    # I1, I2 : Images to match

    # Convert Images to GrayScale
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Detect Features in Both Images
    locs1 = corner_detection(I1_gray, sigma)
    locs2 = corner_detection(I2_gray, sigma)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2

'''
@brief Matches 2 pictures I1, I2 by finding common features in the image
	   But with caching

@param[in] I1 First CV2 Image (not yet in grayscale)
@param[in] I2 First CV2 Image (not yet in grayscale)
@param[in] ratio Parameter to vary briefMatch ratio

@return matches px2 matrix, ith column indices into features in I_i
@return locs1, locs2 Nx2 matrix, (x,y) coordinates of feature points
@return desc1, desc2 Descriptors for caching
'''


def matchPicsCached(I1, I2, ratio=0.67,
					cachedLocs1=None, cachedDesc1=None,
					cachedLocs2=None, cachedDesc2=None):
    # Convert Images to GrayScale
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Detect Features in Both Images
    locs1 = corner_detection(I1_gray) if cachedLocs1 is None else cachedLocs1
    locs2 = corner_detection(I2_gray) if cachedLocs2 is None else cachedLocs2

    # Obtain descriptors for the computed feature locations
	# Use cached data if possible to avoid recomputation
    desc1, locs1 = computeBrief(I1_gray, locs1) \
        if (cachedLocs1 is None and cachedDesc1 is None) \
        else (cachedDesc1, cachedLocs1)

    desc2, locs2 = computeBrief(I2_gray, locs2) \
        if (cachedLocs2 is None and cachedDesc2 is None) \
        else (cachedDesc2, cachedLocs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2, desc1, desc2
