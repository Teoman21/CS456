import numpy as np

def myHoughLineSegments(edgeImg, peaks, rhoScale, thetaScale, distanceThreshold=2, minLength=20):
    """
    Detect line segments in an image using Hough transform peaks.

    Args:
        edgeImg: Binary edge image.
        peaks: List of (rho_idx, theta_idx) tuples representing peaks in Hough space.
        rhoScale: Array mapping rho indices to rho values.
        thetaScale: Array mapping theta indices to theta values.
        distanceThreshold: Maximum distance from a point to a line to be considered an inlier.
        minLength: Minimum length of a line segment to be accepted.

    Returns:
        A list of (x1, y1, x2, y2) tuples representing the endpoints of the detected line segments.
    """

    segments = []
    rows, cols = np.where(edgeImg)

    for rho_idx, theta_idx in peaks:
        rho = rhoScale[rho_idx]
        theta = thetaScale[theta_idx]

        # Find inlier points
        a = np.cos(theta)
        b = np.sin(theta)
        distances = np.abs(a * cols + b * rows - rho)
        inliers = distances < distanceThreshold
        x_coords = cols[inliers]
        y_coords = rows[inliers]

        if len(x_coords) < 2:
            continue

        # Fit line to inliers (robustly)
        try:
            m, c = np.polyfit(x_coords, y_coords, 1)  # y = mx + c
        except:
            continue

        # Find endpoints of the segment
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min = int(m * x_min + c)
        y_max = int(m * x_max + c)

        # Filter segments by length
        segment_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        if segment_length >= minLength:
            segments.append((int(x_min), int(y_min), int(x_max), int(y_max)))

    return segments
