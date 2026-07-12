import argparse
import sys
import cv2
import numpy as np


MODES = ("canny", "sobel", "laplacian", "scharr", "log", "prewitt", "original")


PREWITT_KX = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float32)
PREWITT_KY = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=np.float32)


def nothing(_x):
    """Placeholder callback required by cv2.createTrackbar."""
    pass


def build_controls_window():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 420, 220)
    cv2.createTrackbar("Canny Low", "Controls", 50, 500, nothing)
    cv2.createTrackbar("Canny High", "Controls", 150, 500, nothing)
    cv2.createTrackbar("Sobel Ksize (1-3-5-7)", "Controls", 1, 3, nothing)
    cv2.createTrackbar("Blur Ksize (0=off)", "Controls", 2, 10, nothing)


def get_odd(val, minimum=1):
    """Coerce a trackbar int into a valid odd kernel size."""
    val = max(minimum, val)
    if val % 2 == 0:
        val += 1
    return val


def apply_blur(gray, blur_slider):
    if blur_slider <= 0:
        return gray
    k = get_odd(blur_slider * 2 - 1, minimum=1)  
    return cv2.GaussianBlur(gray, (k, k), 0)


def apply_canny(gray):
    low = cv2.getTrackbarPos("Canny Low", "Controls")
    high = cv2.getTrackbarPos("Canny High", "Controls")
    if low > high:
        low, high = high, low
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_sobel(gray):
    ksize_slider = cv2.getTrackbarPos("Sobel Ksize (1-3-5-7)", "Controls")
    ksize = ksize_slider * 2 + 1
    ksize = min(max(ksize, 1), 7)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def apply_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


def apply_scharr(gray):

    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(scharr_x, scharr_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def apply_log(gray):
 
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


def apply_prewitt(gray):

    gray_f = gray.astype(np.float32)
    prewitt_x = cv2.filter2D(gray_f, -1, PREWITT_KX)
    prewitt_y = cv2.filter2D(gray_f, -1, PREWITT_KY)
    magnitude = cv2.magnitude(prewitt_x, prewitt_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def process_frame(frame, mode, blur_on):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if blur_on:
        blur_slider = cv2.getTrackbarPos("Blur Ksize (0=off)", "Controls")
        gray = apply_blur(gray, blur_slider)

    if mode == "canny":
        result = apply_canny(gray)
    elif mode == "sobel":
        result = apply_sobel(gray)
    elif mode == "laplacian":
        result = apply_laplacian(gray)
    elif mode == "scharr":
        result = apply_scharr(gray)
    elif mode == "log":
        result = apply_log(gray)
    elif mode == "prewitt":
        result = apply_prewitt(gray)
    else:  # original
        result = frame.copy()

    return result


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    return out


def side_by_side(frame, result, mode_label):
    h, w = frame.shape[:2]
    result_resized = cv2.resize(result, (w, h))
    left = label(frame, "Original")
    right = label(result_resized, mode_label.upper())
    return np.hstack([left, right])


def run(source, is_image):
    build_controls_window()
    mode = "canny"
    blur_on = True

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Error: could not read image '{source}'")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: could not open video source '{source}'")
            sys.exit(1)

    print("Controls: [c]anny  [s]obel  [l]aplacian  scha[x]  lo[g]  "
          "[p]rewitt  [o]riginal  [b]lur toggle  [q]uit")

    while True:
        if is_image:
            frame_to_use = frame
        else:
            ret, frame_to_use = cap.read()
            if not ret:
                print("End of video / camera stream.")
                break

        result = process_frame(frame_to_use, mode, blur_on)
        combined = side_by_side(frame_to_use, result, mode)
        cv2.imshow("Edge Detection - press q to quit", combined)

        key = cv2.waitKey(1 if not is_image else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            mode = "canny"
        elif key == ord('s'):
            mode = "sobel"
        elif key == ord('l'):
            mode = "laplacian"
        elif key == ord('x'):
            mode = "scharr"
        elif key == ord('g'):
            mode = "log"
        elif key == ord('p'):
            mode = "prewitt"
        elif key == ord('o'):
            mode = "original"
        elif key == ord('b'):
            blur_on = not blur_on
            print(f"Blur: {'ON' if blur_on else 'OFF'}")

    if not is_image:
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OpenCV Edge Detection Project")
    parser.add_argument("--image", type=str, help="Path to a still image file")
    parser.add_argument("--video", type=str, help="Path to a video file")
    args = parser.parse_args()

    if args.image:
        run(args.image, is_image=True)
    elif args.video:
        run(args.video, is_image=False)
    else:
        run(0, is_image=False)  # default: webcam


if __name__ == "__main__":
    main()