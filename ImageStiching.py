
from cpselect.cpselect import cpselect
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
def convertToKeypoints(ptArray):
    keyPointPt =[];

    for item in ptArray:
        keyPointPt.append(cv.KeyPoint(item[0], item[1], 10) )
    return np.array(keyPointPt);

#Convert control point Map to two arrays of coordinates (x,y)
def controlpointListToArray(controlpointlist):
    pt1List=[];
    pt2List=[];

    for item in controlpointlist:
        pt1List.append([item['img1_x'],item['img1_y']] );
        pt2List.append([item['img2_x'], item['img2_y']]);
    pt1Array=np.float32( np.array(pt1List));
    pt2Array = np.float32(np.array(pt2List));
    return pt1Array,  pt2Array;

 #the two images to panorama
imageLeftPath = "n1.jpg";
imageRightPath = "n2.jpg";

#Select corresponding points manually
controlpointlist = cpselect(imageRightPath,imageLeftPath);
imgLeft= cv.imread(imageLeftPath);
imgRight= cv.imread(imageRightPath);

#conver point Map to point array
leftPtArray, rightPtArray = controlpointListToArray(controlpointlist);

#compute homograpy matrix
H, mask = cv.findHomography(leftPtArray, rightPtArray,cv.RANSAC, 5.0)
rows, cols = imgLeft.shape[0], imgLeft.shape[1];

#Wrap right image according to homography
dst = cv.warpPerspective(imgRight ,H,(rows*2, cols))

#combine two images to panorama
wrappedImg= dst.copy();
dst[0:rows, 0:cols, :] = imgLeft;

#plot various images
plt.subplot(221),plt.imshow(imgLeft,aspect="auto"),plt.title("First Image");
plt.subplot(222),plt.imshow(imgRight,aspect="auto"),plt.title("Second Image");
plt.subplot(223),plt.imshow(wrappedImg,aspect="auto"),plt.title("Warped Image");
plt.subplot(224),plt.imshow(dst,aspect="auto"),plt.title("Panorama");

plt.show()
plt.figure()