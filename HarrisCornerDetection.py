import cv2
import numpy as np

def findCorners(img, window_size, k, thresh):

 # Step 1 in the algorithm Find x and y derivatives dx dy

    kernel = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])


    dx = cv2.filter2D(img, -1, kernel)



    kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    dy = cv2.filter2D(img, -1, kernel)

    #Step 2 in the algorithm
    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2

    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()

    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = int(window_size / 2)   # 1
    r = np.zeros(img.shape)
    # Loop through image and find our corners
    #  print "Finding Corners..."

  #step 3 and 4 in the algorithm
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Calculate sum of squares
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Find determinant and trace, use to get corner response
 #step 5 in the algorithm

            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r[y, x] = det - k*(trace**2)


# If corner response is over threshold, color the point and add to corner list

    cv2.normalize(r, r, 0.0, 1.0, cv2.NORM_MINMAX);
   # r = cv2.erode(r, None)

    cv2.imshow('r', r)
    cv2.waitKey(0)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if r[y, x] < thresh:
                cornerList.append([x, y, r])
                color_img.itemset((y, x, 0), 0)
                color_img.itemset((y, x, 1), 0)
                color_img.itemset((y, x, 2), 255)
    return color_img, cornerList


filename = 'ch.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = np.float32(gray)



color_img, cornerList = findCorners(gray, 3, 0.04, float(0.98))



cv2.imshow('Corners', color_img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()