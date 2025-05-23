import numpy as np
import cv2
# Your own or provided matching function
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

if __name__ == "__main__":
    cv_cover = cv2.imread('../data/cv_cover.jpg')  # template (cover) - UNCOMMENTED!
    cv_desk  = cv2.imread('../data/cv_desk.png')   # target (desk)
    hp_cover = cv2.imread('../data/hp_cover.jpg')   # replacement cover

    # Resize hp_cover to match cv_cover dimensions
    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

    # 2) Match features between cv_cover and cv_desk
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

        # 3) Build correspondences:
    # The keypoints from corner_detection are in (row, col) order.
    # Swap them to (x,y) order before using in the homography.
    x1 = locs2[matches[:, 1], :2][:, ::-1]  # points from cv_desk (target): swap (y,x) -> (x,y)
    x2 = locs1[matches[:, 0], :2][:, ::-1]  # points from cv_cover (template): swap (y,x) -> (x,y)

    # 4) Compute H2to1 with RANSAC so that x1 = H2to1 * x2.
    # This now gives H2to1 mapping cover -> desk.
    H2to1, inliers = computeH_ransac(x1, x2)
    print("[DEBUG] Homography Matrix:\n", H2to1)

    # 5) Warp hp_cover onto cv_desk.
    # Since H2to1 now maps cover -> desk, use it directly:
    compositeImg = compositeH(H2to1, hp_cover, cv_desk, alreadyInverted=True)
    # 6) Save and display
    cv2.imwrite("../results/harrypotter/harrypotter.jpg", compositeImg)
    cv2.imshow("Composite Image", compositeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()