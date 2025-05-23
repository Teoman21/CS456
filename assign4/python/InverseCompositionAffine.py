import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    Q2.3
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
            
    In the inverse compositional approach:
    1. We compute gradients on the template (once) rather than the warped image
    2. We solve for the warp increment that maps I back to T (inverse of traditional approach)
    3. We compose the inverse of this increment with our current warp
    This makes the algorithm much more efficient as many calculations are done only once.
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64
    W = np.eye(3, dtype=npDtype)    # Initialize identity warp
    x1, y1, x2, y2 = rect
    
    # Ensure rectangle coordinates are in the correct order (x1 < x2, y1 < y2)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Crop template image
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    # Create spline interpolations
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)
    
    # Create template region coordinates
    nX, nY = max(1, int(x2 - x1)), max(1, int(y2 - y1))  # Ensure at least 1 sample
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)
    xx, yy = np.meshgrid(coordsX, coordsY)
    
    # Flatten coordinates for easier computation
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    
    # Get template image T(x)
    T = splineT.ev(xx, yy)
    
    # PRE-COMPUTATION STEPS (done once)
    # Compute gradient of template image
    Tx = splineT.ev(xx, yy, dx=1, dy=0)
    Ty = splineT.ev(xx, yy, dx=0, dy=1)
    
    # Compute Jacobian and steepest descent images
    A = np.zeros((xx_flat.size, 6))
    for i in range(xx_flat.size):
        x, y = xx_flat[i], yy_flat[i]
        tx, ty = Tx.flatten()[i], Ty.flatten()[i]
        
        # Steepest descent images = gradient * Jacobian
        # Jacobian for affine is [x y 1 0 0 0; 0 0 0 x y 1]
        A[i, 0] = tx * x
        A[i, 1] = tx * y
        A[i, 2] = tx
        A[i, 3] = ty * x
        A[i, 4] = ty * y
        A[i, 5] = ty
    
    # Compute Hessian (constant for all iterations)
    H = A.T @ A
    # Add small regularization for numerical stability
    H += np.eye(6) * 1e-10
    
    # Iterative alignment
    for _ in range(maxIters):
        # Apply current warp to coordinates
        warped_coords = np.vstack([
            xx_flat, 
            yy_flat,
            np.ones_like(xx_flat)
        ])
        warped_coords = W @ warped_coords
        
        # Reshape back to grid
        xx_prime = warped_coords[0].reshape(xx.shape)
        yy_prime = warped_coords[1].reshape(yy.shape)
        
        # Check if points are inside image boundaries
        valid_points = (xx_prime >= 0) & (xx_prime < width) & (yy_prime >= 0) & (yy_prime < height)
        if not np.all(valid_points):
            # If any points go outside the image, stop iterations
            break
        
        # Warp current image
        I_warped = splineI.ev(xx_prime, yy_prime)
        
        # Compute error image
        error = I_warped - T
        error_flat = error.flatten()
        
        # Compute parameter update (deltaP)
        b = A.T @ error_flat
        deltaP = np.linalg.solve(H, b)
        
        # Create incremental warp matrix
        dW = np.array([
            [1+deltaP[0], deltaP[1], deltaP[2]],
            [deltaP[3], 1+deltaP[4], deltaP[5]],
            [0, 0, 1]
        ])
        
        # Invert incremental warp
        dW_inv = np.linalg.inv(dW)
        
        # Update warp by composition: W ← W ∘ (W)^-1
        W = dW_inv @ W
        
        # Check convergence
        if np.linalg.norm(deltaP) < threshold:
            break

    # Return the 2x3 affine matrix
    M = W[:2, :]
    return M
