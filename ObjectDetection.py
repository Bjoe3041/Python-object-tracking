from ultralytics import YOLO
import cv2
import numpy as np
import mss

# D:\SOFTWARE DEVELOPMENT\Python object tracking\venv\Scripts>python.exe ..\..\ObjectDetection.py
# venv\Scripts\python.exe ObjectDetection.py

# model = YOLO("yolov8n-oiv7.pt")
model = YOLO("runs\\detect\\train6\\weights\\best.pt")

sct = mss.mss()

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 640

cv2.namedWindow("Screen Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Screen Object Detection", 640, 360)

frame_count = 0
results = []

while True:
    frame_count += 1

    sct_img = sct.grab(sct.monitors[1])
    frame = np.array(sct_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    small_frame = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT))

    if frame_count > 5:
        results = model(small_frame, verbose=False)
        frame_count = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(small_frame, (int(x1), int(y1)), (int(x2), int(y2)), (230, 120, 40), 1)
            cv2.putText(small_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 255, 70), 1)

    cv2.imshow("Screen Object Detection", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
