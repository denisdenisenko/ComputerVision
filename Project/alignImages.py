import cv2 as cv
import numpy as np

def align_msaks_and_images(src_image,dst_image, xys, xyd):


    num_cols, num_rows, dim = src_image.shape
    translation_matrix = np.float32([[1, 0, - xys[0]], [0, 1,xys[1] - xyd[1]]])
    src_img_transl = cv.warpAffine(src_image.copy(), translation_matrix, (num_cols, num_rows))


    return src_img_transl



