import cv2
import numpy as np


class Calibration:

    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
    def __init__(self, inner_columns, inner_rows):
        self.columns = inner_columns
        self.rows = inner_rows
        

    def find_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            (self.columns, self.rows)
        )

        if found:
            cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
            )

        return found, corners

    
    def camera_cal_loop(self, camera):
        objpts = []
        imgpts = []
        cap = False

        while True:
            img = camera.get_frame()

            if cap:
                found, corners = self.find_corners(img)
                if found:
                    objpts.append(objp)

                    corners2 = cv2.cornerSubPix(gray,)  
