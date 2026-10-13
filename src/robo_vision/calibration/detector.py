import cv2

# Unfortunately the best way to do this as of now
def detect_charuco(image, detector: cv2.aruco.CharucoDetector):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)

    if charuco_corners is None or len(charuco_corners) < 4:
        return None, None

    return charuco_corners, charuco_ids

# image, LxW
def detect_chessboard(image, board_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        (board_size[1], board_size[0])
    )

    if not found:
        return None

    return corners

def detect__symmetric_circle(image, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, centers = cv2.findCirclesGrid(image, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    if not ret:
        return None

    return centers


def detect_asymmetric_circle(image, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, centers = cv2.findCirclesGrid(image, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if not ret:
        return None
    return centers