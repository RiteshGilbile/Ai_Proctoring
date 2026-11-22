"""Capture webcam face samples for a student and retrain the LBPH model."""

import os
import re
import sys
import time
import subprocess
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "faces-train.py")

CAMERA_INDEX = 0
TARGET_IMAGES = 80
CAPTURE_INTERVAL = 0.30
MIN_FACE_SIZE = 90
BLUR_THRESHOLD = 45.0

print("=" * 55)
print("       DYNAMIC STUDENT FACE REGISTRATION")
print("=" * 55)

if len(sys.argv) > 1:
    student_name = sys.argv[1].strip()
else:
    student_name = input("Enter student name / ID: ").strip()

if not student_name:
    print("Student name cannot be empty.")
    raise SystemExit(1)

# Keep folder names safe
folder_name = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    student_name
).strip("_")

if not folder_name:
    print("Please enter a name containing letters or numbers.")
    raise SystemExit(1)

student_folder = os.path.join(DATASET_DIR, folder_name)

os.makedirs(student_folder, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Could not load the OpenCV face detector.")
    raise SystemExit(1)

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Could not open webcam.")
    print("Close other apps using the camera and try again.")
    raise SystemExit(1)

# Continue numbering so old images are not overwritten
existing = [
    f for f in os.listdir(student_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_number = len(existing)

start_count = image_number
last_capture = 0.0
previous_face = None
captured_this_session = 0

print("\nLook at the camera.")
print("Slowly turn your head left, right, up and down.")
print(f"The program will collect up to {TARGET_IMAGES} new images.")
print("Press Q to finish early.")
print("Keep only one face in the camera frame.\n")

try:

    while captured_this_session < TARGET_IMAGES:

        success, frame = camera.read()

        if not success:
            print("Unable to read from webcam.")
            break

        # Mirror the camera
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(
                MIN_FACE_SIZE,
                MIN_FACE_SIZE
            )
        )

        status = "Keep one face inside the frame"

        if len(detected) == 1:

            x, y, w, h = detected[0]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Add margin around face
            margin_x = int(w * 0.18)
            margin_y = int(h * 0.18)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)

            x2 = min(
                gray.shape[1],
                x + w + margin_x
            )

            y2 = min(
                gray.shape[0],
                y + h + margin_y
            )

            face_crop = gray[
                y1:y2,
                x1:x2
            ]

            if face_crop.size > 0:

                face_crop = cv2.resize(
                    face_crop,
                    (200, 200)
                )

                # Check image sharpness
                blur_score = cv2.Laplacian(
                    face_crop,
                    cv2.CV_64F
                ).var()

                now = time.time()

                # Check whether image is different
                different_enough = (
                    previous_face is None
                    or np.mean(
                        cv2.absdiff(
                            face_crop,
                            previous_face
                        )
                    ) >= 2.0
                )

                if (
                    blur_score >= BLUR_THRESHOLD
                    and different_enough
                    and now - last_capture >= CAPTURE_INTERVAL
                ):

                    image_number += 1

                    image_path = os.path.join(
                        student_folder,
                        f"face_{image_number:04d}.jpg"
                    )

                    if cv2.imwrite(
                        image_path,
                        face_crop
                    ):

                        captured_this_session += 1
                        last_capture = now

                        previous_face = face_crop.copy()

                        status = "Captured clear face image"

                    else:

                        status = "Could not save image"

                elif blur_score < BLUR_THRESHOLD:

                    status = "Image blurry - hold still"

                elif not different_enough:

                    status = "Change your angle slightly"

                else:

                    status = "Capturing..."

        elif len(detected) > 1:

            status = (
                "Multiple faces detected - "
                "keep only the student"
            )

        cv2.putText(
            frame,
            f"Student: {student_name}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"New images: "
            f"{captured_this_session}/{TARGET_IMAGES}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Student Face Registration",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()


print(
    f"\nSaved {captured_this_session} "
    f"new images for {student_name}."
)

total_images = len([
    f for f in os.listdir(student_folder)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

print(
    "Total images in this student's folder:",
    total_images
)

if captured_this_session == 0 and start_count == 0:

    print(
        "No images were captured, "
        "so model training was skipped."
    )

    raise SystemExit(1)


answer = input(
    "Train/update the face recognition model now? (y/n): "
).strip().lower()

if answer == "y":

    print(
        "\nTraining the face recognition model..."
    )

    result = subprocess.run(
        [
            sys.executable,
            TRAIN_SCRIPT
        ],
        cwd=BASE_DIR
    )

    if result.returncode == 0:

        print(
            "\nRegistration and model "
            "training completed."
        )

    else:

        print(
            "\nTraining failed. "
            "Check the messages above."
        )

else:

    print(
        "Images are saved."
    )

    print(
        "Run 'python faces-train.py' "
        "when you are ready to train."
    )