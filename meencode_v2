#  Fixed an issue where no person was detected in the first frame of the test video
#  Added a function to automatically save the output video after testing




from ultralytics import YOLO
import cv2 as cv
print(cv.__version__)
# conda activate yolo312pip

# Load models
pose_model = YOLO("./models/yolo11n-pose.pt")
ball_model = YOLO("./models/yolo11n.pt")

# Video source (0 = webcam)
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("Lateral.mp4")

# 获取视频参数
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv.CAP_PROP_FPS)

# 初始化 VideoWriter
out = cv.VideoWriter(
    "output.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

threshold = 20


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
    # 默认值放在 if 前面
    head = None
    left_wrist = None
    right_wrist = None
    threshold = 20

    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
        kpts = pose_results[0].keypoints.xy
        person = kpts[0]
        head = person[0]
        left_wrist = person[9]
        right_wrist = person[10]

    # 加 None 检查再判断
    if head is not None and left_wrist is not None and right_wrist is not None:
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
    out.write(final_frame)
    cv.imshow("Throw-in Checker", final_frame)

    # Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
