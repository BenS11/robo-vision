import cv2


class Camera:
    def __init__(self, camera_id=0):
        self.id = camera_id
        self.camera = cv2.VideoCapture(camera_id)

        # if camera failed to open, get out
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Failed to capture frame from camera: " + str(self.id))

        return frame

    def release(self):
        self.camera.release()
    