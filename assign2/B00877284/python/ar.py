import numpy as np
import cv2

from loadSaveVid import loadVid, saveVid
from matchPics import matchPics, matchPicsCached

#Import necessary functions

arSourcePath = "../data/ar_source.mov"
bookMovieSourcePath = "../data/book.mov"
resultsdir = "../results"
videoPath = "../results/ar.avi"


"""
Q4.1
"""
def main():
    os.makedirs(resultsdir, exist_ok=True)

    # Preload frames from npy file if possible
    if (os.path.exists("../data/arFrames.npy") and os.path.exists("../data/bookFrames.npy")):
        arFrames = np.load("../data/arFrames.npy")
        bookFrames = np.load("../data/bookFrames.npy")
    # Otherwise load from video and save
    else:
        arFrames = loadVid(arSourcePath)
        bookFrames = loadVid(bookMovieSourcePath)
        np.save("../data/arFrames.npy", arFrames)
        np.save("../data/bookFrames.npy", bookFrames)

    # Load book cover
    bookCover = cv2.imread('../data/cv_cover.jpg')

    compositeFrames = []

    # Start matching frame by frame
    nFrames = min(len(bookFrames), len(arFrames))

    for frameNo in np.arange(startFrame, endFrame, skip):
        try:
            arFrame = arFrames[frameNo]
            bookMovFrame = bookFrames[frameNo]

            # First crop the arFrame's black (top and bottom) bars first

            # Crop the frame (at center) using aspect ratio from the cover picture
            arFrameCropped = cropFrameToCover(arFrame, bookCover)

            # Then with the cropped frame create composite frame
            compositeFrame = overlayFrame(bookCover, bookMovFrame, arFrameCropped)

            compositeFrames.append(compositeFrame)

            # FOR FASTER DEBUGGING: show each image frame; press any key to continue
            cv2.imshow("compositeFrame", compositeFrame)
            cv2.waitKey(0)

        except Exception as e:
            print(f"Failure at frame {frameNo}, saving progress first.")
            print("Failure Reason:", e)


    # Save frames for post processing (can view or save with parseFrames)
    # Would need to write a script (in a new file) if you want to visualize per image frame
    np.save("../results/compositeFrames.npy", np.array(compositeFrames))

    saveVid(videoPath, compositeFrames)


"""
@brief Overlay pre-cropped arFrame on bookMovFrame using bookCover for matches
@param[in] bookCover Cover to overlay over
@param[in] bookMovFrame Frame from the book video (to overlay ON)
@param[in] arFrame Pre-cropped arFrame

@return New composite frame
"""
# NOTE: Because the book cover is repeated, can cache the descriptors and locs of that
cachedLocs1 = None
cachedDesc1 = None
# prevBookMovFrame = None
# prevH = None
def overlayFrame(bookCover, bookMovFrame, arFrameCropped, threshold=10):
    global cachedDesc1, cachedLocs1 #, prevBookMovFrame, prevH

    #Compute features, descriptors and Match features
    # prevBookMovFrame = bookMovFrame
    matches, locs1, locs2, cachedDesc1, _ = matchPicsCached(
        bookCover, bookMovFrame,
        cachedLocs1=cachedLocs1,
        cachedDesc1=cachedDesc1)
    cachedLocs1 = locs1

    # Create set of points (x1, x2) corresponding to various matches
    # NOTE: Points are in (y,x) not (x,y)

    # Find H and inliners using ransac

    # Normalise H for a better fit (like OpenCV does)
    H /= H[2, 2]
    # prevH = H

    # NOTE: AR frame should already be resized during cropping
    # Get composite image
    compositeImg = compositeH(
        H, arFrameCropped, bookMovFrame, alreadyInverted=True)

    prevCompositeImg = compositeImg

    return compositeImg


"""
Crop the frame (at center) using aspect ratio from the cover picture

@param[in] frame Frame of video to crop
@param[in] cover Cover picture
@param[in] Whether or not to resize image to min dimensions before cropping
           Otherwise, performs a center crop

@return Cropped frame
"""
def cropFrameToCover(frame, cover):

    # Resize to fit min dimension before cropping
    frameResized = np.copy(frame)

    # Crop using indexes
    frameCropped = frameResized

    return frameCropped


if __name__ == "__main__":
    main()
