import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

import cv2

"""
Part 1 (Q2): Sparse Reconstruction
"""


def main():

    # 1. Load the two temple images and the points from data/some_corresp.npz
    im1 = cv2.imread("../data/im1.png")
    im2 = cv2.imread("../data/im2.png")
    corresp = np.load("../data/some_corresp.npz")

    # OpenCV uses BGR, while matplotlib uses RGB
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # 2. Run eight_point to compute the fundamental matrix F
    pts1_corresp = corresp["pts1"]
    pts2_corresp = corresp["pts2"]
    F = sub.eight_point(pts1_corresp, pts2_corresp)
    print("Computed Fundamental Matrix F:")
    print(F)

    # This is used for visualization and debugging
    #hlp.displayEpipolarF(im1, im2, F)
    
    # 3. Load points in image 1 from data/temple_coords.npz
    pts1 = np.load("../data/temple_coords.npz")["pts1"]


    # 4. Run epipolar_correspondences to get points in image 2
    pts2 = sub.epipolar_correspondences(im1, im2, F, pts1)
    # This is used for visualization and debugging
    #hlp.epipolarMatchGUI(im1, im2, F)

    # 5. Load intrinsics and compute the camera projection matrix P1
    intrinsics = np.load("../data/intrinsics.npz")
    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    P1 = intrinsics["K1"] @ np.hstack((R1, t1))

    # 6. Compute essential matrix
    E = sub.essential_matrix(F, K1, K2)
    print("Computed Essential Matrix E:")
    print(E)
    # 7. Use camera2 to get 4 camera projection matrices P2
    M2_candidates = hlp.camera2(E)

   # 8-9. Find the correct P2 and triangulate 3D points
    pts3d_list = []
    P2_list = []
    num_front_points = []

    for i in range(M2_candidates.shape[2]):
        # Get the i-th candidate [R|t] matrix
        M2 = M2_candidates[:, :, i]
        
        # Compute projection matrix P2 = K2[R|t]
        P2_candidate = K2 @ M2
        P2_list.append(P2_candidate)
        
        # Triangulate points with this candidate
        pts3d_candidate = sub.triangulate(P1, pts1, P2_candidate, pts2)
        pts3d_list.append(pts3d_candidate)
        
        # Check how many points are in front of both cameras
        # For first camera (points should have positive Z)
        R2 = M2[:, :3]
        t2 = M2[:, 3].reshape(3, 1)
        
        # Transform points to camera coordinates
        front_cam1 = pts3d_candidate[:, 2] > 0
        
        # Camera 2
        pts3d_cam2 = (R2 @ pts3d_candidate.T + t2).T
        front_cam2 = pts3d_cam2[:, 2] > 0
        
        # Count points in front of both cameras
        count = np.sum(front_cam1 & front_cam2)
        num_front_points.append(count)

    # Choose the candidate with the most points in front of both cameras
    best_idx = np.argmax(num_front_points)
    pts3d = pts3d_list[best_idx]
    P2 = P2_list[best_idx]

        
        
    # 10. Compute the reprojection_error
    print("P1 reprojection error:", hlp.reprojection_error(pts3d, pts1, P1))
    print("P2 reprojection error:", hlp.reprojection_error(pts3d, pts2, P2))

    # 11. Scatter plot the correct 3D points
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts3d[:, 0], pts3d[:, 2], -pts3d[:, 1])
    #ax.axis("equal")
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_box_aspect([1,1,1])
    ax.set_xlim(-1, 1)
    ax.set_ylim( 3, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.tight_layout()
    plt.show()

    # 12. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
    R2_t2 = np.linalg.inv(K2) @ P2
    R2 = R2_t2[:, :3]
    t2 = R2_t2[:, 3, np.newaxis]
    np.savez("../data/extrinsics.npz", R1=R1, t1=t1, R2=R2, t2=t2)


if __name__ == "__main__":
    main()
