import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

# COCO class IDs:
# 0 = person
# 67 = cell phone
target_classes = [0, 67]

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Unable to access webcam")
        break

    # Detect people and mobile phones
    results = model(
        frame,
        classes=target_classes,
        conf=0.40,
        verbose=False
    )

    # Draw detection boxes and labels
    output_frame = results[0].plot()

    # Count detected people and phones
    people_count = 0
    phone_count = 0

    if results[0].boxes is not None:

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            if class_id == 0:
                people_count += 1

            elif class_id == 67:
                phone_count += 1

    # Display counts
    cv2.putText(
        output_frame,
        f"People: {people_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output_frame,
        f"Mobile Phones: {phone_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Display output
    cv2.imshow(
        "AI Exam Proctoring - Person and Phone Detection",
        output_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()