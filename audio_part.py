import sounddevice as sd
import numpy as np
import cv2

# Audio settings
sample_rate = 44100
block_size = 1024

audio_level = 0
loud_sound_count = 0
threshold = 0.08

# Function to capture audio
def audio_callback(indata, frames, time_info, status):
    global audio_level, loud_sound_count

    audio_level = np.sqrt(np.mean(indata ** 2))

    if audio_level > threshold:
        loud_sound_count += 1

# Start microphone recording
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=block_size,
    callback=audio_callback
)

stream.start()

print("Audio monitoring started.")
print("Press 'q' to stop.")

while True:
    frame = np.zeros((300, 600, 3), dtype=np.uint8)

    # Display audio level
    cv2.putText(
        frame,
        "Audio Monitoring",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Audio Level: " + str(round(audio_level, 4)),
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Loud Sound Events: " + str(loud_sound_count),
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    if audio_level > threshold:
        cv2.putText(
            frame,
            "WARNING: Loud sound detected!",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            frame,
            "Status: Normal",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Exam Proctoring - Audio Monitor", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

stream.stop()
stream.close()
cv2.destroyAllWindows()