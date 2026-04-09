from ultralytics import YOLO
import cv2 as cv
print(cv.__version__)
# conda activate yolo312pip

# Load models
pose_model = YOLO("./models/yolo11n-pose.pt")
ball_model = YOLO("./models/yolo11n.pt")

# Video source (0 = webcam)
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("videos/test3.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run pose detection
    pose_results = pose_model(frame, verbose=False)

    # Run object detection (ball)
    ball_results = ball_model(frame, verbose=False)

    # Draw pose
    pose_frame = pose_results[0].plot()

    # Draw ball detection on top
    final_frame = ball_results[0].plot(img=pose_frame)

    # Show result
    cv.imshow("Throw-in Checker", final_frame)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
