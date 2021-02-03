import numpy as np
import cv2

def run_main():

    frame= cv2.imread('Picture1.png')
    roi =frame.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0.5)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 6)
    kernel = np.ones((11,11), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
    kernel, iterations=2)
    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2:
            continue
        print(len(cnt))
        if len(cnt) > 100:
            continue
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 1)

    cv2.imshow("Morphological Closing", closing)
    cv2.imshow("Adaptive Thresholding", thresh)
    cv2.imshow('Contours', frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    run_main()
