import cv2
import os

# Check whether the trained model exists
if not os.path.exists("face_model.yml"):
    print("Error: face_model.yml not found.")
    print("Run faces-train.py first.")
    raise SystemExit

# Load trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# Load student names
student_names = {}

if not os.path.exists("student_names.txt"):
    print("Error: student_names.txt not found.")
    raise SystemExit

with open("student_names.txt", "r") as file:
    for line in file:
        line = line.strip()

        if ":" in line:
            student_id, student_name = line.split(":", 1)
            student_names[int(student_id)] = student_name

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

# Open webcam
camera = cv2.VideoCapture(0)

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Unable to access webcam")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    detected_faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in detected_faces:

        # Extract and resize face
        face_image = gray_frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (200, 200))

        # Predict student ID and confidence distance
        student_id, confidence = recognizer.predict(face_image)

        # Lower LBPH distance generally means a closer match.
        # This threshold is only a starting point.
        if confidence < 65 and student_id in student_names:

            name = student_names[student_id]
            label = "Student: " + name
            color = (0, 255, 0)

        else:
            label = "Unknown Person"
            color = (0, 0, 255)

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # Display label
        cv2.putText(
            frame,
            label,
            (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Distance: {confidence:.1f}",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # Display webcam output
    cv2.imshow(
        "AI Exam Proctoring - Face Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()