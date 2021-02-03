import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw




from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.color import rgb2gray

import argparse
import imutils


point,max_val,correlation = 0,0,0

image1 = 'a9.tif'
img_art = cv.imread(image1)

image2 = 'good4.tif'
img_good = cv.imread(image2)

imgn = img_good - img_art

#cv.imshow("a",imgn)

#cv.waitKey(11111111)
#cv.destroyAllWindows()

# Compute the template matching
#cv.matchTemplate(img_art, img_good, correlation, cv.TM_CCORR_NORMED)

# Find the position of the max point
#cv.minMaxLoc(correlation,None, max_val, None, point)





im1 = cv.imread('bumps/a2.tif')
im2 = cv.imread('bumps/good4.tif')

# Convert images to grayscale
image1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
image2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = ssim(image1_gray, image2_gray, full=True)
print("Image similarity:", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] image1 we can use it with OpenCV
diff = (diff * 255).astype("uint8")

#cv.imshow('diff', diff)
#cv.waitKey()



ref_image = rgb2gray(im1)
impaired_image = rgb2gray(im2)

ssim(ref_image, impaired_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)



# template matching


pic = im1
pic2 = cv.imread("bumps/a9.tif")
picGood = cv.imread("bumps/good5.tif")


gray_img = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
gray_img2 = cv.cvtColor(pic2, cv.COLOR_BGR2GRAY)
gray_imgGood = cv.cvtColor(picGood, cv.COLOR_BGR2GRAY)


template = cv.imread("bumps/good4.tif", cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv.matchTemplate(gray_img, template, cv.TM_CCOEFF_NORMED)
result2 = cv.matchTemplate(gray_img2, template, cv.TM_CCOEFF_NORMED)
resultGood = cv.matchTemplate(gray_imgGood, template, cv.TM_CCOEFF_NORMED)


print (result)
print (result2)
print (resultGood)



loc = np.where(result >= 0.82)


for pt in zip(*loc[::-1]):
    cv.rectangle(pic, pt, (pt[0] + 200, pt[1] + 200), (0, 255, 0), 3)

tt = cv.subtract(picGood,pic)
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(tt,kernel,iterations = 5)
dilation = cv.dilate(erosion,kernel,iterations = 5)


opening = cv.morphologyEx(tt, cv.MORPH_OPEN, kernel)

print(tt)
cv.imshow("tt",tt)
cv.imshow("erosion",erosion)
cv.imshow("dilation",dilation)
print(dilation)
temp = np.where(dilation > 0)

print(temp)

## extracting features from substracted and dilated picture


graypic = cv.imread("bumps/good5.tif", cv.IMREAD_GRAYSCALE)

tt2 = cv.cvtColor(dilation,cv.COLOR_BGR2GRAY)
cv.imshow("tt2",tt2)

orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(tt2,None)
# compute the descriptors with ORB
kp, des = orb.compute(tt2, kp)
# draw only keypoints location,not size and orientation
img22 = cv.drawKeypoints(tt2, kp, None, color=(0,255,0), flags=0)
plt.imshow(img22), plt.show()





#cv.imshow("img", pic2)

cv.waitKey(0)
cv.destroyAllWindows()
