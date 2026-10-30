import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Proctoring AI - Grayscale", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()