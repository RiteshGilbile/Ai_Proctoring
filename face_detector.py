import cv2

# Open webcam
camera = cv2.VideoCapture(0)

# Load face detection model
face = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

while True:

    # Capture video
    success, frame = camera.read()

    if not success:
        print("Camera error!")
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces
    faces = face.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces
    for x, y, w, h in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Show number of faces
    cv2.putText(
        frame,
        "Faces: " + str(len(faces)),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Display video
    cv2.imshow("Face Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) == ord('q'):
        break

# Close camera
camera.release()
cv2.destroyAllWindows()