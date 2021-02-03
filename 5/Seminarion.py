#import matplotlib.image as mpimg
#from scipy import ndimage as ndi
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import SobelEdgeDetector as sobel



#importing images

artefact = cv.imread("bumps/artefact.tif")
bad = cv.imread("bumps/bad.tif")
bad18 = cv.imread("bumps/bad18.tif")
bad10 = cv.imread("bumps/bad10.tif")
good = cv.imread("bumps/good.tif")
g15 = cv.imread("bumps/g15.tif")


#substracting 2 images,  good from bad

substarcted = cv.subtract(good,artefact)


imgplot = plt.imshow(substarcted)
plt.show()


#erosion and dilation

kernel = np.ones((5,5),np.uint8)

eroded = cv.erode(substarcted,kernel,iterations = 5 )
dilated = cv.dilate(eroded,kernel,iterations = 5)


imgplot = plt.imshow(dilated)
plt.show()

#converting black to white

dilated[np.where((dilated==[0,0,0]).all(axis=2))] = [255,255,255]


imgplot = plt.imshow(dilated)
plt.show()



gray = cv.cvtColor(dilated, cv.COLOR_BGR2GRAY)

#apply sobel edge detection

grad = sobel.sobel(gray)
threshold, img_mask = cv.threshold(grad,10 ,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(img_mask ,   cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#Chaeck each contour for its shappe feature
for cnt in contours:
    counterOfArtefacts = 0
    approx = cv.approxPolyDP(cnt, .01 * cv.arcLength(cnt, True), True)
    print
    len(approx)
    if len(approx) > 0 :
        area = cv.contourArea(approx)

        #mark only relevant artifacts
        if (area) > 650 :
            counterOfArtefacts = counterOfArtefacts + 1
            (cx, cy), radius = cv.minEnclosingCircle(cnt)
            circleArea = radius * radius * np.pi
            cv.drawContours(artefact, [cnt], 0, (255, 0, 0), 10)
            print(approx)
            print (area)


imgplot = plt.imshow(artefact)
plt.show()


cv.waitKey()

cv.destroyAllWindows()