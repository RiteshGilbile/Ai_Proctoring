import cv2
import mediapipe as mp
import time
import math

# Load face landmark model
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

camera = cv2.VideoCapture(0)

blink_count = 0
eye_closed = False
start_time = time.time()
last_blink_time = time.time()

while True:
    success, frame = camera.read()

    if not success:
        print("Cannot access webcam.")
        break

    # Use the original camera view (not mirrored)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # Approximate eye landmarks
        left_top = landmarks[159]
        left_bottom = landmarks[145]
        left_left = landmarks[33]
        left_right = landmarks[133]

        right_top = landmarks[386]
        right_bottom = landmarks[374]
        right_left = landmarks[362]
        right_right = landmarks[263]

        # Calculate eye opening
        left_eye_height = math.dist(
            [left_top.x, left_top.y],
            [left_bottom.x, left_bottom.y]
        )

        right_eye_height = math.dist(
            [right_top.x, right_top.y],
            [right_bottom.x, right_bottom.y]
        )

        left_eye_width = math.dist(
            [left_left.x, left_left.y],
            [left_right.x, left_right.y]
        )

        right_eye_width = math.dist(
            [right_left.x, right_left.y],
            [right_right.x, right_right.y]
        )

        # Avoid division by zero
        left_ratio = left_eye_height / max(left_eye_width, 0.0001)
        right_ratio = right_eye_height / max(right_eye_width, 0.0001)

        eye_ratio = (left_ratio + right_ratio) / 2

        # Detect a blink using a basic threshold
        if eye_ratio < 0.18:
            if not eye_closed:
                eye_closed = True
        else:
            if eye_closed:
                blink_count += 1
                eye_closed = False
                last_blink_time = time.time()

        cv2.putText(
            frame,
            "Blink Count: " + str(blink_count),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Liveness check: Experimental",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    else:
        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Spoofing - Blink Check", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
landmarker.close()
cv2.destroyAllWindows()