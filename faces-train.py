import cv2
import os
import numpy as np

# Dataset folder
dataset_path = "dataset"

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

# Store student names
student_names = {}

student_id = 0

# Read student folders
for student_name in sorted(os.listdir(dataset_path)):

    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue

    student_names[student_id] = student_name

    print("Reading images for:", student_name)

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)

        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Skipping invalid image:", image_path)
            continue

        # Detect faces
        detected_faces = face_detector.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in detected_faces:

            # Crop the detected face
            face_image = image[y:y+h, x:x+w]

            # Resize for consistency
            face_image = cv2.resize(face_image, (200, 200))

            faces.append(face_image)
            labels.append(student_id)

    student_id += 1

# Train the model
if len(faces) > 0:

    recognizer.train(
        faces,
        np.array(labels, dtype=np.int32)
    )

    # Save trained model
    recognizer.save("face_model.yml")

    # Save student names
    with open("student_names.txt", "w") as file:

        for label, name in student_names.items():
            file.write(str(label) + ":" + name + "\n")

    print("\nTraining completed successfully!")
    print("Total face samples:", len(faces))
    print("Model saved as face_model.yml")
    print("Student names saved as student_names.txt")

else:
    print("No faces found. Check your dataset images and folder structure.")