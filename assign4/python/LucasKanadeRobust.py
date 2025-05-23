import numpy as np
from scipy.interpolate import RectBivariateSpline

def getWeightMatrix(e, a=None, mtype="huber"):
    """
    Compute per‐residual weights for a chosen M‐estimator.
    e     – vector of residuals
    a     – tuning constant (default chosen per standard practice)
    mtype – "huber", "tukey", or "none"
    returns 1-D weight vector w of same length as e
    """
    if mtype == "huber":
        if a is None: a = 1.339
        # weights = min(1, a/|e|)
        w = np.minimum(a/np.maximum(np.abs(e),1e-8), 1.0)
    elif mtype == "tukey":
        if a is None: a = 4.685
        # Tukey’s biweight function
        w = (1 - (e/a)**2)**3
        w = np.clip(w, 0, 1)
    else:
        # no robustness weighting
        w = np.ones_like(e)
    return w

def LucasKanadeRobust(It, It1, rect, mtype="huber"):
    """
    Robust translation‐only Lucas–Kanade.
    Inputs:
      It, It1 – two consecutive grayscale frames
      rect    – [x1,y1,x2,y2] bounding box in It
      mtype   – "huber","tukey" or "none" for the M‐estimator
    Returns:
      p = [dx,dy] the estimated shift from It→It1
    """
    # convergence settings
    threshold = 0.01875
    maxIters  = 100
    p = np.zeros(2, dtype=float)

    # build splines for subpixel sampling
    h,w = It.shape
    xs, ys = np.arange(w), np.arange(h)
    splineT = RectBivariateSpline(xs, ys, It.T)
    splineI = RectBivariateSpline(xs, ys, It1.T)

    # template grid
    x1,y1,x2,y2 = rect
    nX = max(int(np.ceil(x2-x1)),2)
    nY = max(int(np.ceil(y2-y1)),2)
    coordsX = np.linspace(x1, x2, nX)
    coordsY = np.linspace(y1, y2, nY)
    xx, yy = np.meshgrid(coordsX, coordsY)

    # fixed template patch and its mean for brightness normalization
    T = splineT.ev(xx,yy)
    T_mean = T.mean()

    for _ in range(maxIters):
        # warp the grid by current p
        xx_p = xx + p[0]
        yy_p = yy + p[1]
        Iw = splineI.ev(xx_p, yy_p)

        # 1) Brightness scaling: match mean intensities
        Iw_mean = Iw.mean()
        if Iw_mean != 0:
            Iw = Iw * (T_mean / Iw_mean)

        # residuals
        error = (T - Iw).flatten()

        # image gradients at warped positions
        Ix = splineI.ev(xx_p,yy_p,dx=1,dy=0).flatten()
        Iy = splineI.ev(xx_p,yy_p,dx=0,dy=1).flatten()

        # build design matrix A (n×2)
        A = np.vstack((Ix,Iy)).T

        # 2) Robust weights
        w = getWeightMatrix(error, mtype=mtype)

        # weighted normal equations: (AᵀW A) dp = AᵀW b
        Aw = A * w[:,None]
        H  = Aw.T @ A
        g  = Aw.T @ error

        # solve for update dp
        dp, *_ = np.linalg.lstsq(H, g, rcond=None)
        p += dp

        if np.linalg.norm(dp) < threshold:
            break

    return p
