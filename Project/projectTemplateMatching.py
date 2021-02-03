import cv2
def templateMatch (src_img) :
    img_rgb = src_img
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('bumps/cross.png',0)
    # Store width and height of template in w and h
    w, h = template.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    threshold = 0.8
    print(res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_loc)
    return max_loc


#img = cv2.imread('bumps/good4.tif')
#c = templateMatch(img)
#print(c)


