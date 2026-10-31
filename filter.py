import cv2

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Cannot read webcam frame.")
        break

    # Original frame
    original = frame.copy()

    # Grayscale filter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Blur filter
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # Edge detection filter
    edges = cv2.Canny(frame, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Display each filter
    cv2.imshow("Original Webcam", original)
    cv2.imshow("Grayscale Filter", gray)
    cv2.imshow("Blur Filter", blurred)
    cv2.imshow("Edge Detection Filter", edges)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()