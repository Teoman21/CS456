import numpy as np
import cv2

"""
Q3.6
Compute the homography between two sets of points

@param[in] x1  N×2 array of (x,y) points in image1
@param[in] x2  N×2 array of (x,y) points in image2
@return H2to1  3×3 homography that maps x2 -> x1
               i.e., x1 = H2to1 * x2
"""
def computeH(x1, x2):
    n = x1.shape[0]
    A = []
    for i in range(n):
        X1, Y1 = x1[i, 0], x1[i, 1]
        X2, Y2 = x2[i, 0], x2[i, 1]
        # Each point pair yields two rows in A
        A.append([-X2, -Y2, -1,   0,   0,  0,  X1*X2,  X1*Y2,  X1])
        A.append([   0,    0,  0, -X2, -Y2, -1,  Y1*X2,  Y1*Y2,  Y1])

    A = np.array(A)
    # Solve A*h = 0 using SVD (h is 9x1)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]       # the eigenvector with smallest singular value
    H2to1 = h.reshape(3, 3)

    # Normalize so H2to1[2,2] = 1
    H2to1 = H2to1 / H2to1[2, 2]
    return H2to1


"""
Q3.7
Normalize the coordinates to reduce noise before computing H

@param[in] _x1 N×2 array of (x,y) points in image1
@param[in] _x2 N×2 array of (x,y) points in image2
@return H2to1  3×3 homography that maps x2 -> x1
"""
def computeH_norm(_x1, _x2):
    x1 = np.array(_x1)
    x2 = np.array(_x2)

    # 1) Compute centroids
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)

    # 2) Shift points so centroids are at origin
    x1_shifted = x1 - mean1
    x2_shifted = x2 - mean2

    # 3) Scale so that the largest distance from origin is sqrt(2)
    max1 = np.max(np.linalg.norm(x1_shifted, axis=1))
    max2 = np.max(np.linalg.norm(x2_shifted, axis=1))
    s1 = max1 / np.sqrt(2) if max1 != 0 else 1
    s2 = max2 / np.sqrt(2) if max2 != 0 else 1

    x1_norm = x1_shifted / s1
    x2_norm = x2_shifted / s2

    # 4) Construct similarity transforms
    # T1_inv transforms normalized coords -> original coords
    T1_inv = np.array([[s1,   0,   mean1[0]],
                       [ 0,   s1,  mean1[1]],
                       [ 0,    0,         1]])
    # T2 transforms original coords -> normalized coords
    T2 = np.array([[1/s2,   0,   -mean2[0]/s2],
                   [  0,   1/s2, -mean2[1]/s2],
                   [  0,     0,           1]])

    # 5) Compute homography using normalized coords
    H_norm = computeH(x1_norm, x2_norm)

    # 6) Denormalize: H2to1 = T1_inv * H_norm * T2
    H2to1 = T1_inv @ H_norm @ T2
    H2to1 /= H2to1[2, 2]
    return H2to1


"""
Q3.8
RANSAC for homography estimation

@param[in] _x1 N×2 array (image1 coords)
@param[in] _x2 N×2 array (image2 coords)
@param[in] threshold Pixel error squared threshold
@return bestH2to1  Best homography mapping x2->x1
@return bestInliers  N-length binary vector of inliers
"""
def computeH_ransac(_x1, _x2, nSamples=2000, threshold=30):
    x1 = np.array(_x1)
    x2 = np.array(_x2)
    nPoints = x1.shape[0]
    assert nPoints == x2.shape[0]

    bestInlierCount = 0
    bestH2to1 = None
    bestInliers = None

    for i in range(nSamples):
        # Randomly pick 4 matches
        idx = np.random.choice(nPoints, 4, replace=False)
        x1_4 = x1[idx]
        x2_4 = x2[idx]

        # Estimate homography using normalized coords
        H_temp = computeH_norm(x1_4, x2_4)

        # Project all x2 -> x1 using H_temp and compute squared error
        x2_h = np.hstack([x2, np.ones((nPoints, 1))])
        projected = (H_temp @ x2_h.T).T
        projected /= projected[:, [2]]
        errs = np.sum((projected[:, :2] - x1)**2, axis=1)
        inliers = errs < threshold
        countIn = np.sum(inliers)

        if countIn > bestInlierCount:
            bestInlierCount = countIn
            bestH2to1 = H_temp
            bestInliers = inliers.astype(np.uint8)

    print(f"[DEBUG] RANSAC: Best inlier count = {bestInlierCount}/{nPoints}")
    if bestInlierCount < 4:
        print("[WARNING] RANSAC: Too few inliers. Check your matches or try increasing threshold/nSamples.")
    return bestH2to1, bestInliers


"""
Q3.9
Create a composite image after warping 'template' (hp_cover) onto 'img' (cv_desk)
using H2to1, which maps desk->cover. We invert it so that we warp
cover->desk.

@param[in] H2to1   3×3 homography desk->cover
@param[in] template  hp_cover image
@param[in] img       cv_desk image
@return composite   final image with template warped onto img
"""
def compositeH(H2to1, template, img, alreadyInverted=False):
    # If H2to1 is already mapping from cover -> desk, use it directly;
    # otherwise, invert it.
    if alreadyInverted:
        H_warp = H2to1
    else:
        H_warp = np.linalg.inv(H2to1)
    
    h_img, w_img = img.shape[:2]
    print(f"[DEBUG] compositeH: Target image dimensions: {w_img} x {h_img}")

    # Warp the template (cover) to the target image coordinate system.
    warped_template = cv2.warpPerspective(template, H_warp, (w_img, h_img),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0)
    print(f"[DEBUG] compositeH: warped_template min/max = {warped_template.min()}/{warped_template.max()}")

    # Create a single-channel mask from the template's dimensions.
    mask = 255 * np.ones(template.shape[:2], dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, H_warp, (w_img, h_img),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
    print(f"[DEBUG] compositeH: warped_mask unique values = {np.unique(warped_mask)}")

    # Convert warped mask to float and normalize to [0,1] as a weight mask.
    mask_float = warped_mask.astype(np.float32) / 255.0
    if mask_float.ndim == 2:
        mask_float = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)

    # Blend the warped template and the original image using the soft mask.
    composite_img = (warped_template.astype(np.float32) * mask_float +
                     img.astype(np.float32) * (1 - mask_float))
    composite_img = np.clip(composite_img, 0, 255).astype(np.uint8)
    return composite_img