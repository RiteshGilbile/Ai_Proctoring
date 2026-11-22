
import os
import time
import math
import wave
import threading

import cv2
import numpy as np

from alert import check_alerts, show_alert

try:
    import mediapipe as mp
except Exception as error:
    mp = None
    print("MediaPipe unavailable:", error)

try:
    import sounddevice as sd
except Exception as error:
    sd = None
    print("Microphone monitoring unavailable:", error)

try:
    from ultralytics import YOLO
except Exception as error:
    YOLO = None
    print("YOLO unavailable:", error)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ------------------------- CAMERA SETUP -------------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# ------------------------- FACE DETECTION -------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------- FACE RECOGNITION -------------------------
recognizer = None
student_names = {}
face_recognition_status = "Model unavailable"

face_model_path = os.path.join(PROJECT_DIR, "face_model.yml")
names_path = os.path.join(PROJECT_DIR, "student_names.txt")

try:
    if hasattr(cv2, "face") and os.path.exists(face_model_path) and os.path.exists(names_path):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(face_model_path)
        with open(names_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if ":" in line:
                    student_id, student_name = line.split(":", 1)
                    student_names[int(student_id)] = student_name
        face_recognition_status = "Ready"
    else:
        face_recognition_status = "Not configured"
except Exception as error:
    face_recognition_status = "Unavailable"
    print("Face recognition setup error:", error)

# ------------------------- MEDIAPIPE FACE LANDMARKS -------------------------
landmarker = None
landmark_status = "Unavailable"
start_time = time.monotonic()
previous_timestamp = 0

if mp is not None:
    landmark_model_path = os.path.join(PROJECT_DIR, "models", "face_landmarker.task")
    try:
        if os.path.exists(landmark_model_path):
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=landmark_model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=2
            )
            landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            landmark_status = "Ready"
        else:
            landmark_status = "Model missing"
    except Exception as error:
        landmark_status = "Unavailable"
        print("Face landmark setup error:", error)

# ------------------------- YOLO PERSON / PHONE DETECTION -------------------------
yolo_model = None
yolo_status = "Unavailable"
yolo_results = None
frame_number = 0

if YOLO is not None:
    yolo_path = os.path.join(PROJECT_DIR, "yolov8n.pt")
    try:
        if os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
            yolo_status = "Ready"
        else:
            yolo_status = "Model missing"
    except Exception as error:
        yolo_status = "Unavailable"
        print("YOLO setup error:", error)

# ------------------------- MOTION DETECTION -------------------------
previous_gray = None
motion_status = "Starting"

# ------------------------- BLINK / LIVENESS CHECK -------------------------
blink_count = 0
eye_closed = False
liveness_status = "Waiting for face"

# ------------------------- AUDIO MONITORING AND RECORDING -------------------------
sample_rate = 44100
audio_threshold = 0.08
audio_level = 0.0
loud_sound_count = 0
audio_chunks = []
audio_lock = threading.Lock()
audio_stream = None
audio_status = "Unavailable"

# The same microphone stream is used for sound-level monitoring and recording.
def audio_callback(indata, frames, time_info, status):
    global audio_level, loud_sound_count
    if status:
        pass
    samples = indata.astype(np.float32) / 32768.0
    level = float(np.sqrt(np.mean(samples ** 2)))
    audio_level = level
    if level > audio_threshold:
        loud_sound_count += 1
    with audio_lock:
        audio_chunks.append(indata.copy())

if sd is not None:
    try:
        audio_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=audio_callback
        )
        audio_stream.start()
        audio_status = "Active / Recording"
    except Exception as error:
        audio_status = "Microphone unavailable"
        print("Microphone could not start:", error)

# ------------------------- VIDEO RECORDING -------------------------
os.makedirs(os.path.join(PROJECT_DIR, "recorded_video"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "recorded_audio"), exist_ok=True)
video_path = os.path.join(PROJECT_DIR, "recorded_video", "recorded_video.mp4")
audio_path = os.path.join(PROJECT_DIR, "recorded_audio", "recorded_audio.wav")
video_writer = None
recording_status = "Starting"

# ------------------------- HEAD POSE REFERENCE POINTS -------------------------
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# Display mode for the filter feature: 1 original, 2 gray, 3 blur, 4 edges.
display_mode = 1
window_name = "AI Exam Proctoring - All Features"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\nAI Exam Proctoring started in one window.")
print("Press Q / ESC to stop. Press 1-4 to change the displayed filter.\n")

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read webcam frame.")
            break

        frame_number += 1
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        output = frame.copy()

        # Start video recording after the first frame gives its dimensions.
        if video_writer is None:
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (width, height)
            )
            recording_status = "Active" if video_writer.isOpened() else "Video writer error"

        # 1. Motion detection (same logic as motion_detector.py).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if previous_gray is not None:
            difference = cv2.absdiff(previous_gray, blurred_gray)
            _, motion_mask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
            movement_pixels = cv2.countNonZero(motion_mask)
            motion_status = "Movement Detected" if movement_pixels > 5000 else "No Major Movement"
        previous_gray = blurred_gray

        # 2. Face detection and 3. face recognition (same Haar + LBPH approach).
        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80)
    )
        face_labels = []

        # Remove very small/weak detections
        valid_faces = []

        for face in detected_faces:
            x, y, w, h = face
            if w >= 80 and h >= 80:
                valid_faces.append(face)
                
        detected_faces = valid_faces
        for (x, y, w, h) in detected_faces:
            label = "Face detected"
            box_color = (0, 255, 0)
            if recognizer is not None:
                face_image = gray[y:y+h, x:x+w]

                face_image = cv2.resize(
                    face_image,
                    (200, 200)
                )

                # Improve contrast, same approach used in the working faces.py
                face_image = cv2.equalizeHist(face_image)

                student_id, confidence = recognizer.predict(face_image)

                # LBPH uses distance: lower means better match
                if confidence < 85 and student_id in student_names:
                    label = student_names[student_id]
                    box_color = (0, 255, 0)
                else:
                    label = "Unknown Person"
                    box_color = (0, 0, 255)
            cv2.rectangle(output, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(output, label, (x, max(y-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            face_labels.append(label)

        # 4. Face landmarks, 5. eye tracking, 6. mouth opening,
        # 7. head pose and 8. blink/liveness all share one MediaPipe result.
        eye_status = "No face"
        mouth_status = "No face"
        head_status = "No face"
        head_angles = "--"
        if landmarker is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            timestamp_ms = max(timestamp_ms, previous_timestamp + 1)
            previous_timestamp = timestamp_ms
            landmark_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if landmark_result.face_landmarks:
                face_landmarks = landmark_result.face_landmarks[0]

                # Draw all facial landmarks (face_landmark.py).
                for point in face_landmarks:
                    px, py = int(point.x * width), int(point.y * height)
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(output, (px, py), 1, (0, 180, 0), -1)

                # Eye tracker: draw the eight eye points.
                eye_points = [33, 133, 159, 145, 362, 263, 386, 374]
                for point_id in eye_points:
                    point = face_landmarks[point_id]
                    cv2.circle(output, (int(point.x * width), int(point.y * height)), 3, (255, 255, 0), -1)
                eye_status = "Tracking active"

                # Mouth opening detector.
                upper = face_landmarks[13]
                lower = face_landmarks[14]
                mouth_distance = abs(int(lower.y * height) - int(upper.y * height))
                mouth_status = "Open" if mouth_distance > 10 else "Closed"
                cv2.circle(output, (int(upper.x * width), int(upper.y * height)), 4, (0, 255, 255), -1)
                cv2.circle(output, (int(lower.x * width), int(lower.y * height)), 4, (0, 255, 255), -1)

                # Blink count using the same eye aspect ratio idea as face_spoofing.py.
                left_top, left_bottom = face_landmarks[159], face_landmarks[145]
                left_left, left_right = face_landmarks[33], face_landmarks[133]
                right_top, right_bottom = face_landmarks[386], face_landmarks[374]
                right_left, right_right = face_landmarks[362], face_landmarks[263]
                left_height = math.dist((left_top.x, left_top.y), (left_bottom.x, left_bottom.y))
                right_height = math.dist((right_top.x, right_top.y), (right_bottom.x, right_bottom.y))
                left_width = math.dist((left_left.x, left_left.y), (left_right.x, left_right.y))
                right_width = math.dist((right_left.x, right_left.y), (right_right.x, right_right.y))
                eye_ratio = ((left_height / max(left_width, 0.0001)) +
                             (right_height / max(right_width, 0.0001))) / 2
                if eye_ratio < 0.18:
                    if not eye_closed:
                        eye_closed = True
                elif eye_closed:
                    blink_count += 1
                    eye_closed = False
                liveness_status = "Blink detected (experimental)" if blink_count > 0 else "Please blink"

                # Head pose estimation using solvePnP (head_pose_estimation.py).
                landmark_ids = [1, 152, 33, 263, 61, 291]
                image_points = np.array([
                    (face_landmarks[i].x * width, face_landmarks[i].y * height)
                    for i in landmark_ids
                ], dtype=np.float64)
                focal_length = width
                center = (width / 2, height / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                distortion = np.zeros((4, 1), dtype=np.float64)
                pose_ok, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if pose_ok:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                    pitch, yaw, roll = angles[0], angles[1], angles[2]
                    vertical = "Looking Down" if pitch > 12 else "Looking Up" if pitch < -12 else "Head Vertical"
                    horizontal = "Looking Right" if yaw > 15 else "Looking Left" if yaw < -15 else "Facing Forward"
                    head_status = vertical + " / " + horizontal
                    head_angles = f"P:{pitch:.0f} Y:{yaw:.0f}"
            else:
                liveness_status = "Waiting for face"

        # 9. Person and mobile phone detection. Run every third frame to
        # reduce CPU load; reuse the latest result between inference frames.
        people_count = 0
        phone_count = 0
        if yolo_model is not None:
            if frame_number % 3 == 1 or yolo_results is None:
                try:
                    yolo_results = yolo_model(frame, classes=[0, 67], conf=0.40, verbose=False)[0]
                except Exception as error:
                    yolo_status = "Inference error"
                    print("YOLO inference error:", error)
                    yolo_results = None
            if yolo_results is not None and yolo_results.boxes is not None:
                for box in yolo_results.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if class_id == 0:
                        people_count += 1
                        object_label = f"Person {confidence:.2f}"
                        color = (255, 150, 0)
                    else:
                        phone_count += 1
                        object_label = f"Phone {confidence:.2f}"
                        color = (0, 0, 255)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output, object_label, (x1, max(y1-6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # 10. Check proctoring alerts
        check_alerts(people_count, phone_count)

        # 11. Dashboard overlay: all feature statuses are shown in this window.
        panel_height = 250
        overlay = output.copy()
        cv2.rectangle(overlay, (8, 8), (390, panel_height), (15, 15, 15), -1)
        output = cv2.addWeighted(overlay, 0.72, output, 0.28, 0)
        lines = [
            ("AI EXAM PROCTORING - LIVE", (255, 255, 255)),
            (f"Faces: {len(detected_faces)} | {', '.join(face_labels[:2]) if face_labels else 'None'}", (0, 255, 0)),
            (f"Eye tracking: {eye_status}", (255, 255, 0)),
            (f"Mouth: {mouth_status} | Head: {head_status}", (0, 220, 255)),
            (f"Head angles: {head_angles} | Blinks: {blink_count}", (0, 220, 255)),
            (f"Liveness: {liveness_status}", (0, 255, 0) if blink_count else (0, 165, 255)),
            (f"Motion: {motion_status}", (0, 255, 0) if motion_status == "No Major Movement" else (0, 0, 255)),
            (f"People: {people_count} | Phones: {phone_count}", (255, 180, 0)),
            (f"Audio: {audio_level:.3f} | Loud events: {loud_sound_count}", (0, 255, 255)),
            (f"Recording: {recording_status} | Mic: {audio_status}", (0, 255, 0)),
            (f"Recognition: {face_recognition_status} | Landmarks: {landmark_status}", (220, 220, 220)),
            (f"YOLO: {yolo_status} | Filter: {display_mode} (1-4)", (220, 220, 220)),
            ("Q / ESC: stop session", (255, 255, 255))
        ]
        y_text = 29
        for index, (message, color) in enumerate(lines):
            cv2.putText(output, message, (18, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43 if index else 0.52, color, 1 if index else 2, cv2.LINE_AA)
            y_text += 18

        # Display alert banner and warning counter.
        output = show_alert(output, width, height)

        # Save the same combined monitoring view to the video file.
        if video_writer is not None and video_writer.isOpened():
            video_writer.write(output)

        cv2.imshow(window_name, output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            display_mode = int(chr(key))

finally:
    # Stop and release every resource cleanly.
    camera.release()
    if video_writer is not None:
        video_writer.release()
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass
    if landmarker is not None:
        landmarker.close()
    cv2.destroyAllWindows()

    # Save microphone audio captured by the shared input stream.
    with audio_lock:
        saved_audio = list(audio_chunks)
    if saved_audio:
        try:
            audio_array = np.concatenate(saved_audio, axis=0).astype(np.int16)
            with wave.open(audio_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            print("Audio saved:", audio_path)
        except Exception as error:
            print("Could not save audio:", error)
    else:
        print("No audio was recorded.")

    print("Video saved:", video_path if video_writer is not None else "Not recorded")
    print("AI Exam Proctoring session stopped.")
