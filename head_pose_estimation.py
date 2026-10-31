import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Locate the face landmark model
folder_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder_path, "models", "face_landmarker.task")

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure detector
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# Facial landmark indices
landmark_indices = [1, 152, 33, 263, 61, 291]

# Corresponding approximate 3D face points
face_3d_points = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

with FaceLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)
    start_time = time.monotonic()

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Unable to access webcam")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
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

            face = results.face_landmarks[0]

            # Convert normalized landmarks to pixel coordinates
            face_2d_points = []

            for index in landmark_indices:
                x = int(face[index].x * width)
                y = int(face[index].y * height)

                face_2d_points.append([x, y])

            face_2d_points = np.array(
                face_2d_points,
                dtype=np.float64
            )

            # Estimate camera parameters
            focal_length = width
            center = (width / 2, height / 2)

            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            distortion_coefficients = np.zeros((4, 1))

            # Estimate head rotation
            success, rotation_vector, translation_vector = cv2.solvePnP(
                face_3d_points,
                face_2d_points,
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:

                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                    rotation_matrix
                )

                pitch = angles[0] * 360
                yaw = angles[1] * 360

                # Basic head direction classification
                if yaw < -15:
                    direction = "Looking Left"
                elif yaw > 15:
                    direction = "Looking Right"
                elif pitch < -15:
                    direction = "Looking Down"
                elif pitch > 15:
                    direction = "Looking Up"
                else:
                    direction = "Facing Screen"

                cv2.putText(
                    frame,
                    direction,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Display estimated angles
                cv2.putText(
                    frame,
                    f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}",
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
            "AI Exam Proctoring - Head Pose",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()