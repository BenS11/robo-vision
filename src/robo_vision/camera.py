import cv2
import platform
import os
import numpy as np
import logging
from exceptions import CameraOpenError
from pathlib import Path

logger = logging.getLogger(__name__)


class Camera:
    open_cameras = []

    def __init__(self, camera_name, camera_id=0):
        self.id = camera_id
        if camera_id in Camera.open_cameras:
            raise CameraOpenError(f"Camera {camera_id} already open")
        else:
            Camera.open_cameras.append(camera_id)

        if camera_name is None or camera_name == "":
            raise ValueError(f"Camera Name cannot be empty")

        self.camera_name = camera_name
        self.camera = cv2.VideoCapture(camera_id, get_camera_backend())

        # if camera failed to open, get out
        if not self.camera.isOpened():
            raise CameraOpenError(f"Failed to open camera {camera_id}, ID: {camera_id}")

        self.matrix, self.dist = self.__read_calibration()

        # final undistortion parameters
        self.map_x, self.map_y, self.roi, self.new_matrix = None, None, None, None

        test_img = self.get_frame()

        if test_img is None or self.matrix is None or self.dist is None:
            logger.warning("Failed to calibrate camera %s", camera_id)
            self.calibrated = False
        else:
            self.calibrated = True

            h, w = test_img.shape[:2]

            self.new_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.matrix, self.dist, (w, h), 1, (w, h))

            self.map_x, self.map_y = cv2.initUndistortRectifyMap(self.matrix, self.dist, None, self.new_matrix, (w, h),
                                                                 5);

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
        img = self.get_frame()

        if img is None:
            logger.warning("Warning: Failed to get frame from camera %s", self.id)
            return None

        elif not self.is_calibrated():
            logger.warning(
                "Warning: Attempted to get calibrated frame without calibrating camera",
                self.id
            )
            return img

        return cv2.remap(self.get_frame(), self.map_x, self.map_y, cv2.INTER_LINEAR)

    def release(self):
        self.camera.release()
        Camera.open_cameras.remove(self.id)

    def get_name(self):
        return self.camera_name

    def get_calibration_matrix(self):
        if not self.is_calibrated():
            logger.warning("Warning: Attempted to access calibration matrix of uncalibrated camera")

        return self.matrix

    def get_camera_info_path(self):
        # root directory
        project_dir = Path(__file__).resolve().parent.parent.parent
        return project_dir / "cameras" / self.get_name()

    def __read_calibration(self):
        path = os.path.join(self.get_camera_info_path(), "calibration.npz")

        try:
            with np.load(path) as data:
                mtx = data["mtx"]
                dist = data["dist"]

        except (OSError, ValueError, KeyError) as e:
            logger.warning("Could not load camera calibration from %s: %s", path, e)
            return None, None

        if mtx.shape != (3, 3) or dist.size == 0:
            logger.warning("Camera calibration has invalid dimensions: %s", path)
            return None, None

        return mtx, dist

    def is_calibrated(self):
        return self.calibrated


def get_camera_backend():
    system = platform.system()

    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Linux":
        return cv2.CAP_V4L2
    else:
        return cv2.CAP_ANY
