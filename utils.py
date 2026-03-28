import cv2 as cv

cap = cv.VideoCapture(0)

def fetch_frame():
    ret, frame = cap.read()
    if not ret:
        try:
            cap.open(0)
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                return None
        except Exception as e:
            print(f"Camera opening failed: {e}")
            return None
    return frame

def scale_down(frame, scale_percentage):
    if not (1 <= scale_percentage <= 100):
        raise ValueError(f"scale_percentage must be 1-100, got {scale_percentage}")
    width = int(frame.shape[1] * scale_percentage / 100)
    height = int(frame.shape[0] * scale_percentage / 100)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)