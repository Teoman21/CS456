import cv2
import numpy as np
import os
#better aproach create window  when it turn zero 


# Import your pipeline modules.
from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform
from myHoughLines import myHoughLines
from myHoughLineSegments import myHoughLineSegments

# Directories (adjust paths as needed)
datadir    = '../data'
resultsdir = '../results'

# Parameter settings
sigma              = 1.0     # Standard deviation for Gaussian blur (edge detection).  1.0 is a good starting point
threshold          = 0.1       # Threshold for edge magnitude (0.05-0.15).  Adjust based on edge strength. Higher values remove weak edges.
rhoRes             = 1         # Distance resolution in pixels.  1 is standard.
thetaRes           = np.pi/360 # Angular resolution in radians.  np.pi/180 is a good starting point (1 degree).  Lower values increase computational cost.
nLines             = 15        # Maximum number of lines to return.  Adjust based on image complexity.
distanceThreshold  = 10       # Distance threshold for line segment merging.  Adjust based on desired segment length.  Increase to merge nearby segments.
gap_fraction       = 0.1      #Fraction of line length to allow as a gap.




for file in os.listdir(datadir):
    if file.endswith('.jpg'):
        file = os.path.splitext(file)[0]
        img = cv2.imread(f'{datadir}/{file}.jpg')
        if img is None:
            continue
        
        # Convert image to grayscale and normalize.
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img) / 255
        
        # Edge detection using your custom edge filter.
        img_edge = myEdgeFilter(img, sigma)
        # Apply fixed threshold to the normalized edge map.
        img_threshold = np.float32(img_edge > threshold)
        
        # Compute the Hough accumulator.
        img_hough, rhoScale, thetaScale = myHoughTransform(img_threshold, rhoRes, thetaRes)
        
        # Extract line peaks using your custom Hough lines function.
        rhosIdx, thetasIdx = myHoughLines(img_hough, nLines)
        peaks = list(zip(rhosIdx, thetasIdx))
        
        # Now, compute line segments from the edge image and detected peaks.
        segments = myHoughLineSegments(img_threshold, peaks, rhoScale, thetaScale, distanceThreshold)
        
        # Optionally, get OpenCV’s HoughLinesP for comparison.
        houghP_img = np.uint8(255 * img_threshold)
        cv_lines = cv2.HoughLinesP(houghP_img, rhoRes, thetaRes, 30, minLineLength=20, maxLineGap=5)
        
        # Save intermediate images.
        cv2.imwrite(f'{resultsdir}/{file}_edge.png', 255 * (img_edge / (img_edge.max() if img_edge.max()!=0 else 1)))
        cv2.imwrite(f'{resultsdir}/{file}_threshold.png', 255 * img_threshold)
        cv2.imwrite(f'{resultsdir}/{file}_hough.png', 255 * (img_hough / (img_hough.max() if img_hough.max()!=0 else 1)))
        
        # Prepare an image to draw line segments (convert grayscale to BGR).
        img_segments = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Draw your custom line segments in red.
        for (x1, y1, x2, y2) in segments:
            cv2.line(img_segments, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw OpenCV HoughLinesP segments in green for comparison.
        if cv_lines is not None:
            for line in cv_lines:
                coords = line[0]
                cv2.line(img_segments, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
        
        # Save the final output image.
        cv2.imwrite(f'{resultsdir}/{file}_segments.png', img_segments)
