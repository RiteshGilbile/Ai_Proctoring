import cv2
import mediapipe as mp
import time

# Load MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=2
)

# Create face landmarker
with FaceLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot access webcam!")
        exit()

    start_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Camera error!")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image for MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Detect facial landmarks
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Draw facial landmarks
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:

                for point in face_landmarks:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Face Landmarks Detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Display result
        cv2.imshow("Face Landmarks", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()