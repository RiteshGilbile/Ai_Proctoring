import cv2
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "face_model.yml")
NAMES_PATH = os.path.join(BASE_DIR, "student_names.txt")

# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("Error: face_model.yml not found.")
    print("Run faces-train.py first.")
    raise SystemExit

# --------------------------------------------------
# LOAD LBPH MODEL
# --------------------------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

print("Face recognition model loaded.")

# --------------------------------------------------
# LOAD STUDENT NAMES
# --------------------------------------------------

student_names = {}

if not os.path.exists(NAMES_PATH):
    print("Error: student_names.txt not found.")
    raise SystemExit

with open(NAMES_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if ":" in line:
            student_id, student_name = line.split(":", 1)
            student_names[int(student_id)] = student_name

print("Registered students:")

for student_id, student_name in student_names.items():
    print(student_id, "=", student_name)

# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# OPEN WEBCAM
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# --------------------------------------------------
# RECOGNITION THRESHOLD
# --------------------------------------------------

# Lower LBPH distance = better match.
# Start with 85.
RECOGNITION_THRESHOLD = 85

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Unable to access webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
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

    # --------------------------------------------------
    # PROCESS EACH FACE
    # --------------------------------------------------

    for (x, y, w, h) in detected_faces:

        # Extract face
        face_image = gray_frame[y:y+h, x:x+w]

        # Resize to same size used during training
        face_image = cv2.resize(
            face_image,
            (200, 200)
        )

        # Improve contrast
        face_image = cv2.equalizeHist(face_image)

        # Predict
        student_id, distance = recognizer.predict(face_image)

        # --------------------------------------------------
        # RECOGNITION
        # --------------------------------------------------

        if (
            distance < RECOGNITION_THRESHOLD
            and student_id in student_names
        ):

            name = student_names[student_id]

            label = "Student: " + name

            color = (0, 255, 0)

        else:

            label = "Unknown Person"

            color = (0, 0, 255)

        # --------------------------------------------------
        # DRAW FACE BOX
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # --------------------------------------------------
        # DISPLAY NAME
        # --------------------------------------------------

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # --------------------------------------------------
        # DISPLAY DISTANCE
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Distance: {distance:.1f}",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # --------------------------------------------------
    # DISPLAY WINDOW
    # --------------------------------------------------

    cv2.imshow(
        "AI Exam Proctoring - Face Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# CLOSE
# --------------------------------------------------

camera.release()
cv2.destroyAllWindows()