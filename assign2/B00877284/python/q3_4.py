import os
import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

resultsdir = "../results/matchPics"
os.makedirs(resultsdir, exist_ok=True)

matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# Display matched features
saveTo = os.path.join(resultsdir, "matchPics.png")
plotMatches(cv_cover, cv_desk, matches, locs1, locs2, saveTo=saveTo, showImg=True)
