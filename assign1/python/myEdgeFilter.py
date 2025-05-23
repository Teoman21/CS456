import numpy as np
from scipy import signal
from math import ceil, pi
from myImageFilter import myImageFilter

def non_maximum_suppression(grad_mag, angle):
    nms = np.zeros_like(grad_mag)
    rows, cols = grad_mag.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            theta = angle[i, j]
            # Determine neighbors based on gradient direction
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                neighbor1 = grad_mag[i, j+1]
                neighbor2 = grad_mag[i, j-1]
            elif (22.5 <= theta < 67.5):
                # For ~45° gradient, compare with (i-1, j-1) and (i+1, j+1)
                neighbor1 = grad_mag[i-1, j-1]
                neighbor2 = grad_mag[i+1, j+1]
            elif (67.5 <= theta < 112.5):
                neighbor1 = grad_mag[i-1, j]
                neighbor2 = grad_mag[i+1, j]
            elif (112.5 <= theta < 157.5):
                # For ~135° gradient, compare with (i-1, j+1) and (i+1, j-1)
                neighbor1 = grad_mag[i-1, j+1]
                neighbor2 = grad_mag[i+1, j-1]
            # Keep the pixel if it's a local maximum
            if grad_mag[i, j] >= neighbor1 and grad_mag[i, j] >= neighbor2:
                nms[i, j] = grad_mag[i, j]
            else:
                nms[i, j] = 0
    return nms


def myEdgeFilter(img0, sigma):
    # Smooth the image with a Gaussian filter.
    hsize = int(2 * ceil(3 * sigma) + 1)
    gauss_1d = signal.gaussian(hsize, sigma)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    smoothed = myImageFilter(img0, gauss_2d)
    
    # Compute gradients with Sobel filters.
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    grad_x = myImageFilter(smoothed, sobel_x)
    grad_y = myImageFilter(smoothed, sobel_y)
    
    # Compute gradient magnitude and direction.
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * (180 / pi)
    angle[angle < 0] += 180
    
    # --- New: Boost vertical edge responses ---
    # For nearly vertical gradients (angle near 0 or 180), the vertical gradient is small.
    vertical_mask = (angle < 15) | (angle > 165)
    if np.any(vertical_mask):
        # Boost the magnitude by a factor (e.g., 1.5) for vertical edges.
        magnitude[vertical_mask] *= 1.5
    
    # Apply non-maximum suppression.
    edge_nms = non_maximum_suppression(magnitude, angle)
    
    # Normalize the edge map to [0,1] for a fixed threshold to work meaningfully.
    if edge_nms.max() > 0:
        edge_nms = edge_nms / edge_nms.max()
    return edge_nms
