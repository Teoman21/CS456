import os
import scipy.ndimage
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matchPics import matchPics
from helper import plotMatches

resultsdir = "../results/rotTest"

"""
Q3.5
"""
if __name__ == "__main__":
    os.makedirs(resultsdir, exist_ok=True)

    # Read the image and convert to grayscale, if necessary
    originalImg = cv2.imread("../data/cv_cover.jpg")
    rotImg = originalImg.copy()

    # Histogram count for matches
    nMatches = []
    angles = [(i+1)*10 for i in range(36)]

    for i in range(36):
        # Rotate Image
        angle = (i+1)*10
        rotImg = scipy.ndimage.rotate(originalImg, angle, reshape=False)

        # Compute features, descriptors and match features
        matches, locs1, locs2 = matchPics(originalImg, rotImg, ratio=0.67)

        # Update histogram with the actual number of matches
        nMatches.append(len(matches))  # CHANGE: record number of matches

        # Save all results without displaying them
        saveTo = os.path.join(resultsdir, f"rot{angle}.png")
        plotMatches(originalImg, rotImg, matches, locs1, locs2, saveTo=saveTo, showImg=False)

    # Display histogram of match counts vs. rotation angle
    plt.tight_layout()
    plt.bar(x=angles, height=nMatches, width=5)
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Number of Matches")
    plt.title("Histogram of Matches vs. Rotation Angle")
    plt.show()
