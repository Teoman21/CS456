import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    """
    Q3.2
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] M: the Affine warp matrix [2x3 numpy array]
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be useful
    p = np.zeros((6, 1), dtype=npDtype) # OR p = np.zeros((6,1))
    x1, y1, x2, y2 = rect

    # YOUR IMPLEMENTATION HERE
    height, width = It.shape
    _x = np.arange(width)
    _y = np.arange(height)
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    nX = max(int(np.ceil(x_max - x_min)), 2)
    nY = max(int(np.ceil(y_max - y_min)), 2)
    coordsX = np.linspace(x_min, x_max, nX, dtype=npDtype)
    coordsY = np.linspace(y_min, y_max, nY, dtype=npDtype)

    xx, yy = np.meshgrid(coordsX, coordsY)
    T = splineT.ev(xx, yy)

    # Finish after maxIters or [at the end] when deltaP < threshold
    for _ in range(maxIters):

        # Warp image
        #   1. warp coordinates
        m00 = 1.0 + p[0,0]
        m01 = p[1,0]
        m02 = p[2,0]
        m10 = p[3,0]
        m11 = 1.0 + p[4,0]
        m12 = p[5,0]
        xx_prime = m00*xx + m01*yy + m02
        yy_prime = m10*xx + m11*yy + m12

        #   2. warp image (get image from new image locations)
        warpedI = splineI.ev(xx_prime, yy_prime)

        # Compute error image
        error = T - warpedI

        # Compute gradient of warped image
        I_x = splineI.ev(xx_prime, yy_prime, dx=1, dy=0)
        I_y = splineI.ev(xx_prime, yy_prime, dx=0, dy=1)

        # Compute Jacobian and Hessian
        SD = np.stack([
            I_x * xx,
            I_x * yy,
            I_x,
            I_y * xx,
            I_y * yy,
            I_y
        ], axis=0)  # shape (6, nY, nX)
        H = SD.reshape(6, -1) @ SD.reshape(6, -1).T  # 6×6

        # Calculate deltaP
        sd_update = (SD.reshape(6, -1) @ error.flatten()).reshape(6, 1)
        deltaP, _, _, _ = np.linalg.lstsq(H, sd_update, rcond=None)

        # Update p
        p = p + deltaP

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break

    # Reshape the output affine matrix
    M = np.array([
        [1.0 + p[0,0], p[1,0],      p[2,0]],
        [p[3,0],       1.0 + p[4,0], p[5,0]]
    ]).reshape(2, 3)

    return M
