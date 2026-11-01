import os
import subprocess
import sys


while True:
    print("\n========================================")
    print("       AI EXAM PROCTORING SYSTEM      ")
    print("========================================")
    print("1. Face Detection")
    print("2. Motion Detection")
    print("3. Record Video and Audio")
    print("4. Face Landmark Detection")
    print("5. Eye Tracking")
    print("6. Mouth Opening Detection")
    print("7. Head Pose Estimation")
    print("8. Person and Mobile Phone Detection")
    print("9. Face Recognition")
    print("10. Face Spoofing / Blink Check")
    print("11. Audio Monitoring")
    print("12. Image Filters")
    print("0. Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        file_name = "face_detector.py"
    elif choice == "2":
        file_name = "motion_detector.py"
    elif choice == "3":
        file_name = "record_video_audio.py"
    elif choice == "4":
        file_name = "face_landmark.py"
    elif choice == "5":
        file_name = "eye_tracker.py"
    elif choice == "6":
        file_name = "mouth_opening_detector.py"
    elif choice == "7":
        file_name = "head_pose_estimation.py"
    elif choice == "8":
        file_name = "person_and_phone.py"
    elif choice == "9":
        file_name = "faces.py"
    elif choice == "10":
        file_name = "face_spoofing.py"
    elif choice == "11":
        file_name = "audio_part.py"
    elif choice == "12":
        file_name = "filter.py"
    elif choice == "0":
        print("Exiting AI Exam Proctoring System.")
        break
    else:
        print("Invalid choice. Please try again.")
        continue

    if os.path.exists(file_name):
        print("\nStarting", file_name)
        subprocess.run([sys.executable, file_name])
    else:
        print("Error: File not found:", file_name)