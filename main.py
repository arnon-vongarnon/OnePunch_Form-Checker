from ultralytics import YOLO
import cv2 as cv
print(cv.__version__)
# conda activate yolo312pip

# Load models
pose_model = YOLO("./models/yolo11n-pose.pt")
ball_model = YOLO("./models/yolo11n.pt")

# Video source (0 = webcam)
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("videos/test2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Pose detection
    pose_results = pose_model(frame, verbose=False)

    # Ball detection
    ball_results = ball_model(frame, verbose=False)

    # Draw pose
    pose_frame = pose_results[0].plot()

    # Draw ball detection
    final_frame = ball_results[0].plot(img=pose_frame)

    # Check hands
    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
        kpts = pose_results[0].keypoints.xy
        person = kpts[0]

        head = person[0]
        left_wrist = person[9]
        right_wrist = person[10]

    threshold = 20

    if left_wrist[1] < head[1] + threshold and right_wrist[1] < head[1] + threshold:
        text = "Hands OK"
        color = (0, 255, 0)
    else:
        text = "Hands too low"
        color = (0, 0, 255)

    cv.putText(final_frame, text, (50, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    print("Head:", head)
    print("Left wrist:", left_wrist)
    print("Right wrist:", right_wrist)

    # Show result
    cv.imshow("Throw-in Checker", final_frame)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
