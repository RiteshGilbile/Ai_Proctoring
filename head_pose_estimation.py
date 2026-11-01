import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Face Landmarker
model_path = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# Open webcam
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Cannot access webcam.")
    landmarker.close()
    exit()

# Create webcam window
window_name = "AI Exam Proctoring - Head Pose"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

# 3D reference points for the face
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

start_time = cv2.getTickCount()
previous_timestamp = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Cannot read webcam frame.")
        break

    height, width, _ = frame.shape

    # Use the original camera view without mirroring
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Generate increasing video timestamps
    current_timestamp = int(
        (cv2.getTickCount() - start_time)
        * 1000 / cv2.getTickFrequency()
    )

    current_timestamp = max(
        current_timestamp, previous_timestamp + 1
    )

    previous_timestamp = current_timestamp

    # Detect face landmarks
    result = landmarker.detect_for_video(
        mp_image, current_timestamp
    )

    # Camera calibration approximation
    focal_length = width
    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    # Draw center guide
    cv2.line(
        frame, (width // 2, 0),
        (width // 2, height), (255, 255, 0), 1
    )

    cv2.line(
        frame, (0, height // 2),
        (width, height // 2), (255, 255, 0), 1
    )

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # Select six facial landmarks
        landmark_ids = [1, 152, 33, 263, 61, 291]

        image_points = np.array([
            (
                landmarks[i].x * width,
                landmarks[i].y * height
            )
            for i in landmark_ids
        ], dtype=np.float64)

        # Estimate head orientation
        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            angles = cv2.RQDecomp3x3(
                rotation_matrix
            )[0]

            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]

            # Determine head direction
            if pitch > 12:
                vertical_status = "Looking Down"
            elif pitch < -12:
                vertical_status = "Looking Up"
            else:
                vertical_status = "Head Vertical: Normal"

            if yaw > 15:
                horizontal_status = "Looking Right"
            elif yaw < -15:
                horizontal_status = "Looking Left"
            else:
                horizontal_status = "Facing Forward"

            # Draw detected facial points
            for point in image_points:
                cv2.circle(
                    frame,
                    (int(point[0]), int(point[1])),
                    4, (0, 255, 0), -1
                )

            # Display angles and status
            cv2.putText(
                frame,
                "Pitch: " + str(round(pitch, 1)),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                "Yaw: " + str(round(yaw, 1)),
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                vertical_status,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

            cv2.putText(
                frame,
                horizontal_status,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2
            )

    else:
        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()


