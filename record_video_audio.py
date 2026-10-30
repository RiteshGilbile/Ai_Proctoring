import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import os
import numpy as np

# Create separate folders
os.makedirs("recorded_video", exist_ok=True)
os.makedirs("recorded_audio", exist_ok=True)

# Output file paths
video_path = os.path.join("recorded_video", "recorded_video.mp4")
audio_path = os.path.join("recorded_audio", "recorded_audio.wav")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot access webcam!")
    exit()

# Get video dimensions
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
video = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (width, height)
)

# Audio settings
sample_rate = 44100
audio_data = []
stop_audio = threading.Event()

# Record microphone audio
def record_audio():
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16"
        ) as stream:
            while not stop_audio.is_set():
                data, overflowed = stream.read(1024)
                audio_data.append(data.copy())
    except Exception as e:
        print("Audio recording error:", e)

# Start audio thread
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

print("Recording started. Press Q to stop.")

# Record webcam video
while True:
    success, frame = camera.read()

    if not success:
        print("Webcam error!")
        break

    video.write(frame)

    cv2.putText(
        frame,
        "Recording - Press Q to Stop",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Video Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop recording
stop_audio.set()
audio_thread.join()

camera.release()
video.release()
cv2.destroyAllWindows()

# Save audio
if audio_data:
    combined_audio = np.concatenate(audio_data, axis=0)
    write(audio_path, sample_rate, combined_audio)
    print("Audio saved:", audio_path)
else:
    print("No audio recorded.")

print("Video saved:", video_path)
print("Recording completed!")