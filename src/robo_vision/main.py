import cv2
import time


from robo_vision.camera import Camera


def main():
    camera = Camera("Example_Cam")
    frame_count = 0
    start_time = time.time()

    while True:
        frame = camera.get_frame()

        frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Robo Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()