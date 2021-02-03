from Project import alignIm
from Project import projectTemplateMatching
import matplotlib as plt

import cv2 as cv

good = cv.imread("bumps/good5.tif")
a4 = cv.imread("bumps/a4.tif")

xys = projectTemplateMatching.templateMatch(good)
xyd = projectTemplateMatching.templateMatch(a4)

new_image = alignIm.align_images(a4,xys,xyd)

#print(new_image)

#fig, axs = plt.subplots()
#fig.suptitle('Image with Unknown artefact or bad bump', fontsize=16)
#imgplot = plt.imshow(new_image)
#plt.show()



   # print(res)
 #   loc = np.argmax(res)
  #  print(loc)
  #  return loc


#img = cv2.imread('bumps/good4.tif')
#c = templateMatch(img)
#print(c)