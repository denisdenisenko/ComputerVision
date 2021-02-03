#import matplotlib.image as mpimg
#from scipy import ndimage as ndi
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import SobelEdgeDetector as sobel
from Project import projectTemplateMatching
from Project import alignIm


#importing images

#a2 = cv.imread("bumps/a2.tif")
#a9 = cv.imread("bumps/a9.tif")
#bad10 = cv.imread("bumps/bad10.tif")
#bumpsBadArticact = cv.imread("bumps/bumps_bad_articact.tif")
#bumpsBadArticact2 = cv.imread("bumps/bumps_bad_artifact2.tif")

#a4 = cv.imread("bumps/a4.tif")
#bad14 = cv.imread("bumps/bad14.tif")
#bad9 = cv.imread('bumps/bad9.tif')
bad7 = cv.imread('bumps/bad7.tif')
#bad6 = cv.imread('bumps/bad6.tif')
#bad18 = cv.imread('bumps/bad18.tif')


#aligned = cv.imread("aligned.jpg")

good4 = cv.imread("bumps/good4.tif")
#good5 = cv.imread("bumps/good5.tif")

cross = cv.imread("bumps/cross.png")

#Chose the BAD picture to work with

bad = bad7

fig, axs = plt.subplots()
fig.suptitle('bad picture', fontsize=16)
imgplot = plt.imshow(bad)
plt.show()

fig, axs = plt.subplots()
fig.suptitle('cross', fontsize=16)
imgplot = plt.imshow(cross)
plt.show()

#Returning coordinates of best match

badXy = projectTemplateMatching.templateMatch(bad)
godXy = projectTemplateMatching.templateMatch(good4)

#Translation Images

dest = alignIm.align_images(bad,godXy,badXy)

fig, axs = plt.subplots()
fig.suptitle('Image with Unkown Artifact After Translation', fontsize=16)
imgplot = plt.imshow(dest)
plt.show()

y=40
x=40
h=1500
w=2300
cropBad = dest[y:y+h, x:x+w]

fig, axs = plt.subplots()
fig.suptitle('Cropped Image with Unknown Artifact', fontsize=16)
imgplot = plt.imshow(cropBad)
plt.show()


#Choose the pictures to work with

goldenGood = good4

goldenBad = dest

#Cropping the IMAGES

y=40
x=40
h=1500
w=2300
cropGood = goldenGood[y:y+h, x:x+w]

fig, axs = plt.subplots()
fig.suptitle('Reference Image', fontsize=16)
imgplot = plt.imshow(cropGood)
plt.show()


goldenBad = cropBad

goldenGood = cropGood


outFilename = "dest.jpg"
cv.imwrite(outFilename, goldenBad)


#substracting 2 images,  good from bad

substarcted = cv.subtract(goldenGood,goldenBad)



"""
fig, axs = plt.subplots()
fig.suptitle('Image with Unknown artefact or bad bump', fontsize=16)
imgplot = plt.imshow(goldenBad)
plt.show()"""

fig, axs = plt.subplots()
fig.suptitle('Substracted', fontsize=16)
imgplot = plt.imshow(substarcted)
plt.show()


#erosion and dilation

kernel = np.ones((13,13),np.uint8)
eroded = cv.erode(substarcted,kernel,iterations = 1 )

fig, axs = plt.subplots()
fig.suptitle('Eroded', fontsize=16)
imgplot = plt.imshow(eroded)
plt.show()


kernel = np.ones((5,5),np.uint8)
dilated = cv.dilate(eroded,kernel,iterations = 3)

fig, axs = plt.subplots()
fig.suptitle('Dilated after erosion', fontsize=16)
imgplot = plt.imshow(dilated)
plt.show()


gray = cv.cvtColor(dilated, cv.COLOR_BGR2GRAY,0)

fig, axs = plt.subplots()
fig.suptitle('converted to gray', fontsize=16)
imgplot = plt.imshow(gray)
plt.show()

#converting from black to white

"""   # Inversion to invert the colors of the image b/w

dilated[np.where((dilated==[0,0,0]).all(axis=2))] = [255,255,255]


fig, axs = plt.subplots()
fig.suptitle('Inverted', fontsize=16)
imgplot = plt.imshow(dilated)
plt.show()

"""





#apply sobel edge detection


grad = sobel.sobel(gray)
threshold, img_mask = cv.threshold(grad,10 ,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

fig, axs = plt.subplots()
fig.suptitle('thresh', fontsize=16)
imgplot = plt.imshow(grad)
plt.show()

outFilename = "grad.jpg"
cv.imwrite(outFilename, grad)

contours, hierarchy = cv.findContours(img_mask ,   cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#Check each contour for its shappe feature

counterOfArtefacts = 0
Arrays = None
for cnt in contours:
    approx = cv.approxPolyDP(cnt, .01 * cv.arcLength(cnt, True), True)
    len(approx)
    if len(approx) > 0 :
        area = cv.contourArea(approx)

        #mark only relevant artifacts
        if (area) > 0 :
            counterOfArtefacts = counterOfArtefacts + 1
            (cx, cy), radius = cv.minEnclosingCircle(cnt)
            circleArea = radius * radius * np.pi
            cv.drawContours(goldenBad, [cnt], 0, (255, 0, 0), 10)

            


fig = plt.figure()
fig.suptitle('Defect Recognition on Pattern sample', fontsize=14, fontweight='bold')

"""
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)
ax.text(1250, 1400, 'Nubmer of defects found = ' + str(counterOfArtefacts) , style='italic',
       bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 5})

ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))
"""
imgplot = plt.imshow(goldenBad)
plt.show()


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


#TEMPLATE MATCHING FOR BAD BUMPS DETECTION


"""   # Additional feature for bad bumbs recognition

img_rgb = goldenBad
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

fig, axs = plt.subplots()
fig.suptitle('BadToMatch', fontsize=16)
imgplot = plt.imshow(img_gray)
plt.show()


template = cv.imread('bumps/goodBump.png',0)
w, h = template.shape[::-1]


fig, axs = plt.subplots()
fig.suptitle('template', fontsize=16)
imgplot = plt.imshow(template)
plt.show()

res = cv.matchTemplate(img_gray,template,cv.TM_CCORR_NORMED)
threshold = 0.672
print(res)

loc = np.where(res >= threshold)
#print(loc)
loc = np.array(loc)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h),(255,10,0),1)
    #print(pt)

fig, axs = plt.subplots()
fig.suptitle('Plotted', fontsize=16)
imgplot = plt.imshow(img_rgb)
plt.show()

"""
