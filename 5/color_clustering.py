import cv2
import numpy as np
import matplotlib.pyplot as plt
colors =[[253,159,159],[0,255,0],[0,255,255],[0,0,255],[255,0,0],[255,255,0],[0,100,100]]
from sklearn.cluster import KMeans
image = cv2.imread("untitled.png")
Z = image.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)

ret,label,center = cv2.kmeans(Z,6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image

center = np.uint8(center)
colors = np.uint8(colors)
res = colors[label.flatten()]
segmented_image = res.reshape((image.shape))
clustered = segmented_image.astype(np.uint8)
labels_map = label.reshape((image.shape[0], image.shape[1]))
labels = np.unique(labels_map)
colors =[[255,0,0],[255,255,0],[0,255,255],[0,255,0],[0,0,255],[0,100,100],[100,0,100]]
c = 0
for i in range(len(labels)):
    mask = np.zeros_like(labels_map)
    mask[labels_map == labels[i]]= 255
    plt.imshow(mask)
    plt.show()
    contours, hierarchy = cv2.findContours\
        (mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image,(x,y),(x+w,y+h),colors[c], 4)
        if (c == 6) :
            c = 0
    c = c + 1



cv2.imshow("res",image)
cv2.waitKey(0)


