import cv2
import numpy as np
colors =[[253,159,159],[0,255,0],[0,255,255],[0,0,255],[255,0,0],[255,255,0],[0,100,100]]
from sklearn.cluster import KMeans
image = cv2.imread('candies.jpg')
Z = image.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)

ret,label,center = cv2.kmeans(Z,6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image

center = np.uint8(center)
colors= np.uint8(colors)
res = colors[label.flatten()]
segmented_image = res.reshape((image.shape))
clustered=  segmented_image.astype(np.uint8);


cv2.imshow("res",clustered)
cv2.waitKey(0)
