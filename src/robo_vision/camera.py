import cv2
import os
import numpy as np
import glob
import logging

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, camera_name, camera_id=0):
        self.id = camera_id
        self.camera = cv2.VideoCapture(camera_id)

        # if camera failed to open, get out
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        if camera_name == None or camera_name == "":
            self.release()
            raise RuntimeError(f"Camera Name cannot be empty string")

        self.camera_name = camera_name


        self.matrix, self.distortion = self.__read_calibration()
        self.calibrated = False if self.matrix is None else True

   

    
    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            logger.warning(
                "Warning: Failed to capture frame from camera %s ",
                self.id
            )
            return None

        return frame

    def get_corrected_frame(self):
        if not self.is_calibrated():
            logger.warning(
                "Warning: Attempted to get calibrated frame without calibrating camera",
                self.id
            )
            return self.get_frame()

        # TODO: Impliment undistortion
        return self.get_frame

    def release(self):
        self.camera.release()

    def get_name(self):
        return self.camera_name

    def get_calibration_matrix(self):
        if not self.is_calibrated():
            print("Warning: Attempted to access calibration matrix of uncalibrated camera")
        return self.matrix

    def __read_calibration(self):
        path = "calibrations/" + self.camera_name
        if (os.path.isdir(path)):
            files = glob.glob(os.path.join(path, "*.npz"))
            if (len(files) != 1):
                raise RuntimeError("Found 0 or more than one npz file in " + path)
            loaded = np.load(files[0])
            return (loaded["mtx"], loaded["dist"])

        else:
            return None, None
        
    def is_calibrated(self):
        return self.calibrated
    

    