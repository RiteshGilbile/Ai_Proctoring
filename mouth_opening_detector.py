import cv2
import mediapipe as mp
import time
import os

# Get model path
folder_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder_path, "models", "face_landmarker.task")

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure face landmark detector
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Approximate mouth landmark indices
upper_lip = 13
lower_lip = 14
left_mouth = 78
right_mouth = 308

# Create detector
with FaceLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)
    start_time = time.monotonic()

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Unable to access webcam")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if results.face_landmarks:

            for face in results.face_landmarks:

                # Get upper and lower lip coordinates
                upper = face[upper_lip]
                lower = face[lower_lip]

                upper_x = int(upper.x * width)
                upper_y = int(upper.y * height)

                lower_x = int(lower.x * width)
                lower_y = int(lower.y * height)

                # Calculate mouth opening distance
                mouth_distance = abs(lower_y - upper_y)

                # Display lip landmarks
                cv2.circle(
                    frame,
                    (upper_x, upper_y),
                    4,
                    (0, 255, 0),
                    -1
                )

                cv2.circle(
                    frame,
                    (lower_x, lower_y),
                    4,
                    (0, 255, 0),
                    -1
                )

                # Simple threshold for mouth opening
                if mouth_distance > 10:
                    status = "Mouth Open"
                    color = (0, 0, 255)
                else:
                    status = "Mouth Closed"
                    color = (0, 255, 0)

                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    "Mouth Distance: " + str(mouth_distance),
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "AI Exam Proctoring - Mouth Detector",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
