import cv2
import numpy as np
import os

LEFT_IMG   = "../data/Pic2.jpeg"
RIGHT_IMG  = "../data/Pic3.jpeg"
OUTPUT_IMG = "../results/panorama/panorama.jpg"

def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find columns that contain ANY non‑black pixel
    cols = np.where((gray > 0).any(axis=0))[0]
    if cols.size:
        return img[:, cols[0]:cols[-1]+1]
    return img



def stitch_two(left_path, right_path):
    img1 = cv2.imread(left_path)
    img2 = cv2.imread(right_path)

    orb = cv2.ORB_create(5000)
    kp1, d1 = orb.detectAndCompute(img1, None)
    kp2, d2 = orb.detectAndCompute(img2, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < 30:
        raise RuntimeError("Too few matches")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    panorama = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    panorama[0:img1.shape[0], 0:img1.shape[1]] = img1
    return crop_black(panorama)

if __name__ == "__main__":
    pano = stitch_two(LEFT_IMG, RIGHT_IMG)
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    cv2.imwrite(OUTPUT_IMG, pano)
    print(f"Saved panorama → {OUTPUT_IMG}")
