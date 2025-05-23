"""
Programming Assignment 3
Submission Functions
"""

# import packages here
import helper
import numpy as np
import scipy.linalg
from scipy.signal import convolve2d

"""
Q2.1 Eight Point Algorithm
   [I] pts1 -- points in image 1 (Nx2 matrix)
       pts2 -- points in image 2 (Nx2 matrix)
       M -- scalar value computed as max(H, W)
   [O] F -- the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2):
    # Normalize points using helper.normalizePoints (provided in helper.py)
    pts1_norm, T1 = helper.normalize_points(pts1)
    pts2_norm, T2 = helper.normalize_points(pts2)
    
    N = pts1_norm.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1,      y1,      1]
    
    # Solve Af = 0 using SVD and reshape to get F
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1]
    F_norm = f.reshape(3, 3)
    
    # Enforce rank 2 constraint on the normalized F
    U, S, Vt = np.linalg.svd(F_norm)
    S[-1] = 0
    F_norm_rank2 = U @ np.diag(S) @ Vt

    # Optionally refine F using helper.refineF
    F_refined = helper.refineF(F_norm_rank2, pts1_norm, pts2_norm)
    
    # Unnormalize: if pts_norm = T*pts then F = T2^T * F_refined * T1
    F = T2.T @ F_refined @ T1
    return F



"""
Q2.2 Epipolar Correspondences
   [I] im1 -- image 1 (H1xW1 matrix)
       im2 -- image 2 (H2xW2 matrix)
       F -- fundamental matrix from image 1 to image 2 (3x3 matrix)
       pts1 -- points in image 1 (Nx2 matrix)
   [O] pts2 -- points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    
    window_size = 15
    half_win = window_size // 2
    pts2 = []
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    H, W = im2.shape[:2]
    # For each point in image 1, search along its epipolar line in image 2.
    for (x, y) in pts1:
        x_h = np.array([x, y, 1])
        # Compute the epipolar line l in im2: l = F*x
        l = F @ x_h
        a, b, c = l
        norm_factor = np.sqrt(a**2 + b**2)
        if norm_factor != 0:
            a, b, c = a / norm_factor, b / norm_factor, c / norm_factor
        # Extract a patch around (x, y) in im1
        x_int = int(round(x))
        y_int = int(round(y))
        patch1 = im1[max(y_int-half_win, 0):min(y_int+half_win+1, im1.shape[0]),
                     max(x_int-half_win, 0):min(x_int+half_win+1, im1.shape[1])]
        best_score = float('inf')
        best_x2, best_y2 = 0, 0
        # Search over candidate x2 positions near x (±30 pixels)
        for x2 in range(max(x_int-30, half_win), min(x_int+30, W-half_win)):
            if abs(b) < 1e-6:
                continue
            y2 = int(round((-c - a * x2) / b))
            if y2 - half_win < 0 or y2 + half_win >= H:
                continue
            patch2 = im2[y2-half_win:y2+half_win+1, x2-half_win:x2+half_win+1]
            if patch1.shape != patch2.shape:
                continue
            ssd = np.sum((patch1 - patch2) ** 2)
            if ssd < best_score:
                best_score = ssd
                best_x2, best_y2 = x2, y2
        pts2.append([best_x2, best_y2])
    return np.array(pts2)


"""
Q2.3 Essential Matrix
   [I] F -- the fundamental matrix (3x3 matrix)
       K1 -- camera matrix 1 (3x3 matrix)
       K2 -- camera matrix 2 (3x3 matrix)
   [O] E -- the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2.0
    E_fixed = U @ np.diag([s, s, 0]) @ Vt
    # Normalize the essential matrix to have unit Frobenius norm
    E_fixed = E_fixed / np.linalg.norm(E_fixed)
    return E_fixed






"""
Q2.4 Triangulation
   [I] P1 -- camera projection matrix 1 (3x4 matrix)
       pts1 -- points in image 1 (Nx2 matrix)
       P2 -- camera projection matrix 2 (3x4 matrix)
       pts2 -- points in image 2 (Nx2 matrix)
   [O] pts3d -- 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))
    for i in range(num_points):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        #slighty different şn the lecture
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])
        # Solve for the 3D point using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pts3d[i] = X[:3] / X[-1]
    return pts3d



"""
Q3.1 Image Rectification
   [I] K1 K2 -- camera matrices (3x3 matrix)
       R1 R2 -- rotation matrices (3x3 matrix)
       t1 t2 -- translation vectors (3x1 matrix)
   [O] M1 M2 -- rectification matrices (3x3 matrix)
       K1p K2p -- rectified camera matrices (3x3 matrix)
       R1p R2p -- rectified rotation matrices (3x3 matrix)
       t1p t2p -- rectified translation vectors (3x1 matrix)
"""

def rectify_pair(K1, K2, R1, R2, t1, t2):
    # 1. Compute the optical centers for each camera
    c1 = (-np.linalg.inv(R1) @ t1).ravel()
    c2 = (-np.linalg.inv(R2) @ t2).ravel()

    # 2. Compute the new rotation matrix R_tilde:
    # (a) New x-axis: r1 = (c1 - c2)/||c1 - c2||
    baseline = c1 - c2
    r1 = baseline / np.linalg.norm(baseline)
    
    # (b) New y-axis: use the z unit vector from the old left camera (R1[2,:]) as the arbitrary unit vector.
    r2 = np.cross(R1[2, :].ravel(), r1)
    r2 = r2 / np.linalg.norm(r2)
    
    # (c) New z-axis: r3 = cross(r2, r1)
    r3 = np.cross(r2, r1)
    r3 = r3 / np.linalg.norm(r3)
    
    # Form the new rotation matrix by stacking r1, r2, r3 as rows.
    R_tilde = np.vstack((r1, r2, r3))
    # Set new rectified rotations for both cameras.
    R1p = R_tilde.copy()
    R2p = R_tilde.copy()

    # 3. New intrinsic parameters: K1' = K2' = K2.
    K1p = K2.copy()
    K2p = K2.copy()

    # 4. Compute the new translation vectors: t' = -R_tilde * c.
    t1p = -R_tilde @ c1
    t2p = -R_tilde @ c2

    # 5. Compute the rectification matrices:
    #    M_i = (K'_i * R'_i)(K_i * R_i)⁻¹   for i=1,2.
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    
    W = int(2 * K1[0, 2])  # width ~ 2 * principal point x-coordinate
    H = int(2 * K1[1, 2])  # height ~ 2 * principal point y-coordinate

    # Define the 4 corners in homogeneous coordinates.
    corners = np.array([[0,    W,  W,  0],
                        [0,    0,  H,  H],
                        [1,    1,  1,  1]])
    
    # Warp the corners with each rectification matrix.
    warped_corners1 = M1 @ corners
    warped_corners1 /= warped_corners1[2, :]
    warped_corners2 = M2 @ corners
    warped_corners2 /= warped_corners2[2, :]
    
    # Compute the minimum x and y values from both warped sets.
    min_x1 = np.min(warped_corners1[0, :])
    min_y1 = np.min(warped_corners1[1, :])
    min_x2 = np.min(warped_corners2[0, :])
    min_y2 = np.min(warped_corners2[1, :])
    
    offset_x = -min(min_x1, min_x2)
    offset_y = -min(min_y1, min_y2)
    
    # Build a translation offset matrix.
    T_offset = np.array([[1, 0, offset_x],
                         [0, 1, offset_y],
                         [0, 0, 1]])
    
    # Update both rectification matrices with the offset.
    M1 = T_offset @ M1
    M2 = T_offset @ M2

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p




"""
Q3.2 Disparity Map
   [I] im1 -- image 1 (H1xW1 matrix)
       im2 -- image 2 (H2xW2 matrix)
       max_disp -- scalar maximum disparity value
       win_size -- scalar window size value
   [O] dispM -- disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # Ensure images are in float32 format for proper difference calculations
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    
    # Compute half window size (assumes win_size is odd)
    w = (win_size - 1) // 2
    
    # Create a kernel of ones for summing the squared differences over the window
    kernel = np.ones((win_size, win_size), dtype=np.float32)
    
    # Get image dimensions (assuming both images have the same size)
    H, W = im1.shape
    
    # Create a cost volume to hold matching costs for each disparity candidate (0 to max_disp)
    cost_volume = np.zeros((H, W, max_disp + 1), dtype=np.float32)
    
    # Loop over each candidate disparity d
    for d in range(0, max_disp + 1):
        # Shift im2 right by d pixels.
        # np.roll shifts the image, but we must zero out the wrapped-around part to invalidate it.
        shifted_im2 = np.roll(im2, d, axis=1)
        if d > 0:
            shifted_im2[:, :d] = 0  # set invalid region to zero (or high cost later)
        
        # Compute squared difference between im1 and shifted im2
        diff_squared = (im1 - shifted_im2) ** 2
        
        # Sum differences over a win_size x win_size window using convolution.
        # 'same' mode ensures cost image size stays HxW.
        cost = convolve2d(diff_squared, kernel, mode="same", boundary="symm")
        
        # Store the resulting cost in the cost volume at disparity index d
        cost_volume[:, :, d] = cost

    # For each pixel, choose the disparity that minimizes the cost over disparity candidates.
    dispM = np.argmin(cost_volume, axis=2).astype(np.uint16)
    
    return dispM

"""
Q3.2.3 Depth Map
   [I] dispM -- disparity map (H1xW1 matrix)
       K1 K2 -- camera matrices (3x3 matrix)
       R1 R2 -- rotation matrices (3x3 matrix)
       t1 t2 -- translation vectors (3x1 matrix)
   [O] depthM -- depth map (H1xW1 matrix)
"""

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # Compute the optical centers for each camera:
    # c = -R⁻¹ t (flatten to a 1D vector)
    c1 = (-np.linalg.inv(R1) @ t1).ravel()
    c2 = (-np.linalg.inv(R2) @ t2).ravel()
    
    # Compute the baseline b as the Euclidean distance between the optical centers.
    b = np.linalg.norm(c1 - c2)
    
    # Focal length f is assumed to be given by K1(1,1), which is K1[0,0] in Python (0-indexed)
    f = K1[0, 0]
    
    # Initialize the depth map as a float array of same shape as dispM.
    depthM = np.zeros_like(dispM, dtype=np.float32)
    
    # Avoid division by zero: For pixels with non-zero disparity, compute depth using the formula.
    nonzero = (dispM != 0)
    depthM[nonzero] = (b * f) / dispM[nonzero]
    
    return depthM


"""
Q4.1 Camera Matrix Estimation
   [I] x -- 2D points (Nx2 matrix)
       X -- 3D points (Nx3 matrix)
   [O] P -- camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # x: 2D points, expected as 2×N, but test script passes N×2 -> fix if needed.
    if x.shape[0] != 2:
        x_mat = x.T  # Now x_mat is 2×N
    else:
        x_mat = x

    # X: 3D points, expected as 3×N, but test script passes N×3 -> fix if needed.
    if X.shape[0] != 3:
        X_mat = X.T  # Now X_mat is 3×N
    else:
        X_mat = X

    # Number of correspondences (assume same N from both x_mat and X_mat)
    N = x_mat.shape[1]
    
    # Convert 3D points to homogeneous coordinates: now X_h is a 4×N array.
    X_h = np.vstack((X_mat, np.ones((1, N))))
    
    # Build the linear system A such that A * p = 0, where p is a 12x1 vector.
    A = []
    for i in range(N):
        xi = x_mat[0, i]
        yi = x_mat[1, i]
        Xi = X_h[:, i]  # a 4-vector for the ith point
        # Two equations for each correspondence
        # Equation 1: -X_i^T, 0^T, xi * X_i^T
        # Equation 2:  0^T, -X_i^T, yi * X_i^T
        row1 = np.hstack( (-X_h[:, i].T, np.zeros(4), xi * X_h[:, i].T) )
        row2 = np.hstack( (np.zeros(4), -X_h[:, i].T, yi * X_h[:, i].T) )
        A.append(row1)
        A.append(row2)
    
    A = np.vstack(A)   # A is (2N) x 12

    # Solve for p using SVD; the solution is the right singular vector corresponding to the smallest singular value.
    U, S, Vt = np.linalg.svd(A)
    p = Vt[-1, :]     # p is shape (12,)
    
    # Reshape vector p to a 3×4 camera matrix.
    P = p.reshape(3, 4)
    
    return P


"""
Q4.2 Camera Parameter Estimation
   [I] P -- camera matrix (3x4 matrix)
   [O] K -- camera intrinsics (3x3 matrix)
       R -- camera extrinsics rotation (3x3 matrix)
       t -- camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # Compute SVD on P to recover camera center c
    # Solve P * [c;1] = 0. The last column of V (or Vt[-1]) gives the homogeneous camera center.
    U, S, Vt = np.linalg.svd(P)
    c_h = Vt[-1, :]  # homogeneous camera center (4-vector)
    c = c_h[:3] / c_h[3]  # convert to inhomogeneous (3x1) vector
    
    # Extract the M matrix from P, where M = K*R = P[:, :3]
    M = P[:, :3]
    
    # Run RQ decomposition on M.
    K, R = scipy.linalg.rq(M)
    
    # Adjust signs: ensure the diagonal of K is positive and det(R) > 0.
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
            R[i, :] *= -1
    if np.linalg.det(R) < 0:
        R = -R
    
    # Compute translation: t = -R * c.
    t = -R @ c
    
    return K, R, t
