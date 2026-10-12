import cv2 as cv
import numpy as np
import glob
import camera


def calibrate(camera, inner_columns, inner_rows):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []





