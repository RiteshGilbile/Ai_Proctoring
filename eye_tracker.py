import cv2
import mediapipe as mp
import time
import os

# Get the model path
folder_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder_path, "models", "face_landmarker.task")

# MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure the face landmark detector
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Eye landmark points
eye_points = [
    33, 133, 159, 145,
    362, 263, 386, 374
]

# Create the detector
with FaceLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    camera = cv2.VideoCapture(0)

    start_time = time.monotonic()

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Unable to access webcam")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Calculate timestamp in milliseconds
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        # Detect facial landmarks
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if results.face_landmarks:

            for face in results.face_landmarks:

                # Draw eye landmarks
                for point in eye_points:

                    landmark = face[point]

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Eye Tracking Active",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
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

        # Display webcam output
        cv2.imshow(
            "AI Exam Proctoring - Eye Tracker",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()