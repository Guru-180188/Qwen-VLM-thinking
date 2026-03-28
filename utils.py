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
