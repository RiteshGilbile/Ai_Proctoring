import cv2

# Open webcam
camera = cv2.VideoCapture(0)

# Read the first frame
success, previous_frame = camera.read()

if not success:
    print("Cannot access webcam!")
    camera.release()
    exit()

# Convert the first frame to grayscale
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce noise
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera error!")
        break

    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

    # Compare current frame with previous frame
    difference = cv2.absdiff(previous_gray, current_gray)

    # Highlight areas where movement occurred
    _, threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # Count changed pixels
    movement_pixels = cv2.countNonZero(threshold)

    # Display movement status
    if movement_pixels > 5000:
        status = "Movement Detected"
    else:
        status = "No Major Movement"

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Motion Detector", frame)
    cv2.imshow("Movement Areas", threshold)

    # Save current frame for the next comparison
    previous_gray = current_gray

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()