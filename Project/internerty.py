import cv2
import numpy as np


MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.5


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg
if __name__ == '__main__':

  # Read reference image
  refFilename = "bumps/good4.tif"
  imFilename =  "bumps/a4.tif"
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg = alignImages(im, imReference)

  # Write aligned image to disk.
  outFilename = "aligned.jpg"
  cv2.imwrite(outFilename, imReg)



alined = cv2.imread("aligned.jpg", 0)
alined = alined[:, :280]

b = cv2.imread("bumps/good4.tif", 0 )
b = b[:, :280]

print (alined.shape)
print (b.shape)

diff = cv2.absdiff(alined, b)
cv2.imwrite("diff.png", diff)

threshold = 25
alined[np.where(diff >  threshold)] = 255
alined[np.where(diff <= threshold)] = 0

cv2.imwrite("threshold.png", diff)