import cv2
import cv2.aruco as aruco


# currently will just support and test in 36h11
TAG_FAMILIES = {
    "16h5": aruco.DICT_APRILTAG_16h5,
    "25h9": aruco.DICT_APRILTAG_25h9,
    "36h10": aruco.DICT_APRILTAG_36h10,
    "36h11": aruco.DICT_APRILTAG_36h11,
}

def create_detector(family:str = "36h11"):
    try:
        id = TAG_FAMILIES[family]
    except KeyError:
        raise ValueError(f"Unsupported tag family: {family}") from None

    dictionary = aruco.getPredefinedDictionary(id)
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)