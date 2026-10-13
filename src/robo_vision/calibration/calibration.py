import logging

import cv2 as cv
import numpy as np

from robo_vision.camera import Camera


logger = logging.getLogger(__name__)

# Number of internal chessboard corners (columns, rows)
CHECKERBOARD_SIZE = (9, 6)

# Physical width of one checkerboard square, in meters
SQUARE_SIZE = 1.0


def calibrate_camera(camera_name):
    """Calibrate a camera using images of a chessboard.

    Returns:
        tuple: (camera_matrix, distortion_coefficients, rms_error)
    """
    camera_directory = Camera.get_camera_directory_path_with_name(
        camera_name
    )

    images_directory = camera_directory / "calibration" / "calibration_images"
    output_path = camera_directory / "calibration" / "calibration_data.npz"

    if not images_directory.is_dir():
        raise FileNotFoundError(
            f"Calibration image directory does not exist: "
            f"{images_directory}"
        )

    # Collect images with supported extensions.
    images = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(images_directory.glob(extension))

    images.sort()

    if not images:
        raise FileNotFoundError(
            f"No calibration images found in {images_directory}"
        )

    # Termination criteria for subpixel corner refinement.
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # Prepare the known 3D coordinates of the chessboard corners.
    object_points_template = np.zeros(
        (CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3),
        dtype=np.float32,
    )

    object_points_template[:, :2] = (
        np.mgrid[
            0:CHECKERBOARD_SIZE[0],
            0:CHECKERBOARD_SIZE[1],
        ].T.reshape(-1, 2)
        * SQUARE_SIZE
    )

    # Corresponding 3D world points and 2D image points.
    object_points = []
    image_points = []

    image_size = None
    successful_images = 0

    for image_path in images:
        image = cv.imread(str(image_path))

        if image is None:
            logger.warning("Could not read image: %s", image_path)
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        current_image_size = gray.shape[::-1]

        if image_size is None:
            image_size = current_image_size
        elif current_image_size != image_size:
            logger.warning(
                "Skipping image with inconsistent dimensions: %s",
                image_path,
            )
            continue

        # Detect the internal chessboard corners.
        found, corners = cv.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            None,
        )

        if not found:
            logger.warning(
                "Chessboard corners not found in %s",
                image_path,
            )
            continue

        # Refine corner locations to subpixel accuracy.
        refined_corners = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        object_points.append(object_points_template.copy())
        image_points.append(refined_corners)

        successful_images += 1

        # Display detected corners for visual verification.
        display_image = image.copy()
        cv.drawChessboardCorners(
            display_image,
            CHECKERBOARD_SIZE,
            refined_corners,
            found,
        )

        cv.imshow("Calibration", display_image)
        cv.waitKey(500)

        logger.info("Detected corners in %s", image_path.name)

    cv.destroyAllWindows()

    if successful_images == 0:
        raise ValueError(
            "No chessboard corners were detected in any valid image."
        )

    logger.info(
        "Detected chessboard corners in %d of %d images.",
        successful_images,
        len(images),
    )

    # Estimate the camera's intrinsic matrix and lens distortion.
    rms_error, camera_matrix, distortion_coefficients, rvecs, tvecs = (
        cv.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )
    )


    # Calculate the mean reprojection error across all calibration images.
    total_error = 0.0
    total_corners = 0

    for i in range(len(object_points)):
        projected_points, _ = cv.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            distortion_coefficients,
        )

        # Convert both arrays to matching (N, 2) float32 arrays.
        observed = image_points[i].reshape(-1, 2).astype(np.float32)
        projected = projected_points.reshape(-1, 2).astype(np.float32)

        # Sum Euclidean distances between corresponding corners.
        error = cv.norm(observed, projected, cv.NORM_L2)

        total_error += error
        total_corners += len(observed)

    # Average Euclidean reprojection error per corner, in pixels.
    mean_reprojection_error = total_error / total_corners


    # Save the calibration results.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        matrix=camera_matrix,
        distortion=distortion_coefficients,
        rms_error=rms_error,
        mean_reprojection_error=mean_reprojection_error,
        image_size=np.array(image_size),
        checkerboard_size=np.array(CHECKERBOARD_SIZE),
        square_size=SQUARE_SIZE,
    )

    logger.info("Camera matrix:\n%s", camera_matrix)
    logger.info("Distortion coefficients:\n%s", distortion_coefficients)
    logger.info("RMS calibration error: %.4f", rms_error)
    logger.info(
        "Mean reprojection error: %.4f pixels",
        mean_reprojection_error,
    )
    logger.info("Calibration saved to %s", output_path)

    return camera_matrix, distortion_coefficients, rms_error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    calibrate_camera("precision_3490_builtin")
