import numpy as np
import cv2

def myHoughLines(H, nLines):
    # Smooth the accumulator for better peak detection.
    H_float = H.astype(np.float32)
    H_smoothed = cv2.GaussianBlur(H_float, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    H_dilated = cv2.dilate(H_smoothed, kernel, iterations=1)
    accumulator = H_dilated.copy()
    peaks = []
    nbh_size = 9    # Neighborhood size for non-maximum suppression.
    half_nbh = nbh_size // 2
    while True:
        idx = np.argmax(accumulator)
        peak_val = accumulator.flat[idx]
        # Adaptive threshold: break if peak is less than 20% of current max (with floor 10).
        if peak_val < max(10, 0.2 * accumulator.max()):
            break
        rho_idx, theta_idx = np.unravel_index(idx, accumulator.shape)
        peaks.append((rho_idx, theta_idx))
        # Zero out the neighborhood around the detected peak.
        rho_min = max(rho_idx - half_nbh, 0)
        rho_max = min(rho_idx + half_nbh + 1, accumulator.shape[0])
        theta_min = max(theta_idx - half_nbh, 0)
        theta_max = min(theta_idx + half_nbh + 1, accumulator.shape[1])
        accumulator[rho_min:rho_max, theta_min:theta_max] = 0
        if len(peaks) >= nLines:
            break
    if peaks:
        rhos, thetas = zip(*peaks)
        return np.array(rhos), np.array(thetas)
    else:
        return np.array([]), np.array([])
