import cv2
import os
import shutil
from robo_vision.camera import Camera

GRID_SIZE = (9, 6)

# Default port, run detect_camera to find cameras on other ports
CAMERA_PORT = 0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

def capture_calibration(camera):

    #Whether to draw all captured corners (T) or just the most recent set (F)
    draw_collected_corners = True

    image_count = 0

    # Create the directory for writing captured images to -->
    camera_dir = camera.get_camera_info_path()

    image_dir = ""
    while True:
        try:
            image_dir = os.path.join(camera_dir, "calibration", "calibration_images")
            os.makedirs(image_dir)
            break
        except FileExistsError:
            yes = input(f"Overwrite saved images in {image_dir}? Y/N?:")
            if yes.lower() == "y":
                shutil.rmtree(image_dir)
            else:
                print("Can only only maintain one calibration")
                return
    # <-- 

    
    collected_corners = []

    
    while True:
        img = camera.get_frame()
        if img is  not None:
            save_copy = img.copy()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(gray, GRID_SIZE, None)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                cv2.drawChessboardCorners(img, GRID_SIZE, corners, True)

            # Draw the other collected corners if we want to, we will collect the most recent capture later
            if draw_collected_corners:
                for collected in collected_corners:
                    cv2.drawChessboardCorners(img, GRID_SIZE, collected, True)
       


        # Display the result
        cv2.imshow("Collected Corners", img)

        # Detect a key being pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            image_count += 1
            cv2.imwrite(image_dir + f"/img{image_count}.jpg", save_copy)
            
            #save the corners for display
            collected_corners.append(corners)
        # toggle drawing corners
        elif key == ord("t"):
            draw_collected_corners = not draw_collected_corners
        # quit
        elif key == ord("q"):
            if image_count > 0:
                print("Wrote images to " + image_dir)
            else:
                print("could not write images")
            return


if __name__ == "__main__":
    capture_calibration(Camera("precision_3490_builtin"))
