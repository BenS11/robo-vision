import cv2



def find_camera_indices(upperbound=25):
    indices = []
    prev_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(0)

    for i in range(upperbound):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)

        if cam.isOpened():
            indices.append(i)
        cam.release()

    cv2.utils.logging.setLogLevel(prev_level)
    print(f"Checked cameras range 0-{upperbound}")
    return indices


if __name__ == "__main__":
    print("Found camera(s) at:" + str(find_camera_indices()))
