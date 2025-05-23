import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    """
    Q3.1
        [I] It: Template image
            It1: Current image
            rect: Current position of the object
                (top left, bottom right coordinates: x1, y1, x2, y2)
        [O] p: movement vector dx, dy
    """

    # Set up the threshold
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64    # Might be useful
    # p := dx, dy
    p = np.zeros(2, dtype=npDtype)  # OR p = np.zeros(2)
    x1, y1, x2, y2 = rect

    # Crop template image
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    # This returns a class object; note the swap/transpose
    # Use spline.ev() for getting values at locations
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    nX, nY = int(x2 - x1), int(y2 - y1)
    coordsX = np.linspace(x1, x2, nX, dtype=npDtype)
    coordsY = np.linspace(y1, y2, nY, dtype=npDtype)

    # YOUR IMPLEMENTATION STARTS HERE
    xx, yy = np.meshgrid(coordsX, coordsY)
    T = splineT.ev(xx, yy)

    for _ in range(maxIters):

        # Warp image
        #   1. warp coordinates
        xx_prime = xx + p[0]
        yy_prime = yy + p[1]

        #   2. warp image (get image from new image locations)
        warpedI = splineI.ev(xx_prime, yy_prime)

        # Compute error image
        error = T - warpedI

        # Compute gradient of warped image
        I_x = splineI.ev(xx_prime, yy_prime, dx=1, dy=0)
        I_y = splineI.ev(xx_prime, yy_prime, dx=0, dy=1)

        # Compute Hessian; It is a special case
        H = np.array([[np.sum(I_x**2),    np.sum(I_x * I_y)],
                      [np.sum(I_x * I_y), np.sum(I_y**2)]])

        # Calculate deltaP
        sd_update = np.array([np.sum(I_x * error), np.sum(I_y * error)])
        deltaP, _, _, _ = np.linalg.lstsq(H, sd_update, rcond=None)

        # Update p
        p = p + deltaP

        # Continue unless below threshold
        if np.linalg.norm(deltaP) < threshold:
            break
    # YOUR IMPLEMENTATION ENDS HERE

    return p
