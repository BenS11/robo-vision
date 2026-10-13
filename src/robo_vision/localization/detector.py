import cv2

from robo_vision.calibration.find_cameras import find_camera_indices
from robo_vision.camera import Camera
from robo_vision.localization import tags

class AprilTagDetector:

    def __init__(self, tag_family="36h11"):
        self.detector = tags.create_detector(tag_family)

    def detect(self, image):
        if image is None:
            return [], None, []

        corners, ids, rejected = self.detector.detectMarkers(image)

        if ids is not None:
            return corners, ids, rejected
        else:
            return None, None, None


    # utility function for drawing all candidates and images
    def detect_and_draw(self, image):
        corners, ids, rejected = self.detect(image)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0,255,0))

        if rejected:
            cv2.aruco.drawDetectedMarkers(image, rejected, borderColor=(0,0,255))

        return image

if __name__ == "__main__":
    camera = Camera("Builtin")
    detector = AprilTagDetector()
    while cv2.waitKey(1) & 0xFF != ord('q'):
        img = detector.detect_and_draw(camera.get_frame())

        cv2.imshow("Detections: ", img)

