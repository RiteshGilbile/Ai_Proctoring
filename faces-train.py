import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
NAMES_PATH = os.path.join(BASE_DIR, "student_names.txt")

# Check LBPH availability
if not hasattr(cv2, "face"):
    print("Error: cv2.face is unavailable.")
    print("Install opencv-contrib-python.")
    raise SystemExit(1)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

training_faces = []
training_labels = []
student_names = {}

student_id = 0

# Check dataset folder
if not os.path.isdir(DATASET_DIR):
    print("Dataset folder not found:", DATASET_DIR)
    raise SystemExit(1)

# Read every student folder
for student_name in sorted(os.listdir(DATASET_DIR)):

    student_folder = os.path.join(DATASET_DIR, student_name)

    if not os.path.isdir(student_folder):
        continue

    if student_name.startswith("."):
        continue

    print("\nReading student:", student_name)

    student_names[student_id] = student_name

    image_count = 0

    # Read all saved face images
    for image_name in sorted(os.listdir(student_folder)):

        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(
            student_folder,
            image_name
        )

        # Read image as grayscale
        face_image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if face_image is None:
            print("Skipping:", image_name)
            continue

        # Our registration program already saved
        # cropped face images, so DO NOT detect another face here.
        face_image = cv2.resize(
            face_image,
            (200, 200)
        )

        # Improve contrast
        face_image = cv2.equalizeHist(face_image)

        training_faces.append(face_image)
        training_labels.append(student_id)

        image_count += 1

    print("Usable images:", image_count)

    if image_count > 0:
        student_id += 1
    else:
        student_names.pop(student_id, None)


# Check training data
if len(training_faces) == 0:

    print("\nNo training images found.")
    print("Run register_student.py first.")

    raise SystemExit(1)


# Train LBPH model
print("\nTraining model...")

recognizer.train(
    training_faces,
    np.array(
        training_labels,
        dtype=np.int32
    )
)


# Save trained model
recognizer.save(MODEL_PATH)


# Save student names
with open(
    NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    for student_id, student_name in student_names.items():

        file.write(
            f"{student_id}:{student_name}\n"
        )


print("\n==========================================")
print("       TRAINING COMPLETED SUCCESSFULLY")
print("==========================================")

print("Total images:", len(training_faces))
print("Total students:", len(student_names))

print("\nModel saved:")
print(MODEL_PATH)

print("\nStudent names saved:")
print(NAMES_PATH)

print("\nRegistered students:")

for student_id, student_name in student_names.items():

    print(
        student_id,
        "->",
        student_name
    )