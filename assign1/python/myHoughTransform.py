import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    rows, cols = Im.shape
    
    # Define rho and theta ranges
    max_rho = np.sqrt(rows**2 + cols**2)
    rhoScale = np.arange(-max_rho, max_rho, rhoRes)
    thetaScale = np.arange(0, np.pi, thetaRes)
    
    # Initialize accumulator array
    accumulator = np.zeros((len(rhoScale), len(thetaScale)), dtype=np.uint64)  # Use uint64

    # Precompute cosines and sines
    cos_theta = np.cos(thetaScale)
    sin_theta = np.sin(thetaScale)
    
    # Find edge pixel indices
    y_idxs, x_idxs = np.nonzero(Im)  # Use y_idxs and x_idxs
    
    # Vote in the accumulator array
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for j in range(len(thetaScale)):
            rho = x * cos_theta[j] + y * sin_theta[j]
            rho_idx = np.argmin(np.abs(rhoScale - rho))
            accumulator[rho_idx, j] += 1
            
    return accumulator, rhoScale, thetaScale
