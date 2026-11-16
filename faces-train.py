import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_model.yml"
)

NAMES_PATH = os.path.join(
    BASE_DIR,
    "student_names.txt"
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():

    print(
        "Error: OpenCV Haar face detector "
        "could not be loaded."
    )

    raise SystemExit(1)


if not hasattr(cv2, "face"):

    print(
        "Error: cv2.face is unavailable."
    )

    print(
        "Install opencv-contrib-python."
    )

    raise SystemExit(1)


recognizer = (
    cv2.face.LBPHFaceRecognizer_create()
)

training_faces = []
training_labels = []

student_names = {}

student_id = 0


if not os.path.isdir(DATASET_DIR):

    print(
        "Dataset folder not found:",
        DATASET_DIR
    )

    raise SystemExit(1)


# Read every student folder
for student_name in sorted(
    os.listdir(DATASET_DIR)
):

    student_folder = os.path.join(
        DATASET_DIR,
        student_name
    )

    if (
        not os.path.isdir(student_folder)
        or student_name.startswith(".")
    ):
        continue

    student_names[student_id] = student_name

    student_samples = 0

    print(
        "Reading images for:",
        student_name
    )

    # Read every image
    for image_name in sorted(
        os.listdir(student_folder)
    ):

        if not image_name.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        ):
            continue

        image_path = os.path.join(
            student_folder,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            print(
                "Skipping invalid image:",
                image_path
            )

            continue

        detected_faces = (
            face_detector.detectMultiScale(
                image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )
        )

        if len(detected_faces) == 0:

            print(
                "No face found, skipping:",
                image_name
            )

            continue

        # Use largest detected face
        x, y, w, h = max(
            detected_faces,
            key=lambda box:
            box[2] * box[3]
        )

        face_image = image[
            y:y + h,
            x:x + w
        ]

        face_image = cv2.resize(
            face_image,
            (200, 200)
        )

        training_faces.append(
            face_image
        )

        training_labels.append(
            student_id
        )

        student_samples += 1

    print(
        "  Usable face samples:",
        student_samples
    )

    if student_samples > 0:

        student_id += 1

    else:

        student_names.pop(
            student_id,
            None
        )


if len(training_faces) == 0:

    print(
        "No usable face samples found."
    )

    print(
        "Register students first."
    )

    raise SystemExit(1)


if len(set(training_labels)) < 1:

    print(
        "No student labels available "
        "for training."
    )

    raise SystemExit(1)


# Train LBPH
recognizer.train(
    training_faces,
    np.array(
        training_labels,
        dtype=np.int32
    )
)


# Save trained model
recognizer.save(
    MODEL_PATH
)


# Save student names
with open(
    NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    for label, name in student_names.items():

        file.write(
            f"{label}:{name}\n"
        )


print(
    "\nTraining completed successfully!"
)

print(
    "Total face samples:",
    len(training_faces)
)

print(
    "Number of students:",
    len(student_names)
)

print(
    "Model saved as:",
    MODEL_PATH
)

print(
    "Student names saved as:",
    NAMES_PATH
)