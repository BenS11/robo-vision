import cv2
import numpy as np


class Calibration:
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

