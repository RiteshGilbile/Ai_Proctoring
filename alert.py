import time
import cv2


# -----------------------------------
# ALERT SETTINGS
# -----------------------------------

WARNING_DISPLAY_SECONDS = 3.0

# Detection must remain stable for this many checks
REQUIRED_CONSECUTIVE_DETECTIONS = 5

# Minimum time between two alerts
ALERT_COOLDOWN_SECONDS = 5.0


# -----------------------------------
# ALERT VARIABLES
# -----------------------------------

warning_count = 0
warning_message = ""
warning_until = 0.0

multiple_person_frames = 0
phone_frames = 0

last_alert_time = 0.0
last_alert_type = ""


def check_alerts(people_count, phone_count):

    global warning_count
    global warning_message
    global warning_until

    global multiple_person_frames
    global phone_frames

    global last_alert_time
    global last_alert_type


    current_time = time.monotonic()


    # -----------------------------------
    # MULTIPLE PERSON DETECTION
    # -----------------------------------

    if people_count >= 2:
        multiple_person_frames += 1
    else:
        multiple_person_frames = 0


    # -----------------------------------
    # PHONE DETECTION
    # -----------------------------------

    if phone_count >= 1:
        phone_frames += 1
    else:
        phone_frames = 0


    # -----------------------------------
    # DETERMINE CONFIRMED WARNING
    # -----------------------------------

    current_warning_type = ""
    current_warning_text = ""


    # Multiple persons has priority
    if multiple_person_frames >= REQUIRED_CONSECUTIVE_DETECTIONS:

        current_warning_type = "multiple_persons"

        current_warning_text = (
            "WARNING: MULTIPLE PERSONS DETECTED"
        )


    elif phone_frames >= REQUIRED_CONSECUTIVE_DETECTIONS:

        current_warning_type = "phone"

        current_warning_text = (
            "WARNING: MOBILE PHONE DETECTED"
        )


    # -----------------------------------
    # CREATE ALERT
    # -----------------------------------

    if current_warning_type:

        time_since_last_alert = (
            current_time - last_alert_time
        )


        # Only create a new alert if cooldown has passed
        if time_since_last_alert >= ALERT_COOLDOWN_SECONDS:

            warning_count += 1

            warning_message = current_warning_text

            warning_until = (
                current_time
                + WARNING_DISPLAY_SECONDS
            )

            last_alert_time = current_time

            last_alert_type = current_warning_type

            print(
                f"ALERT {warning_count}: "
                f"{warning_message}"
            )


    return (
        warning_count,
        warning_message,
        warning_until
    )


def show_alert(output, width, height):

    # -----------------------------------
    # WARNING BANNER
    # -----------------------------------

    if (
        warning_message
        and time.monotonic() < warning_until
    ):

        banner_height = 58

        cv2.rectangle(
            output,
            (0, height - banner_height),
            (width, height),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            warning_message,
            (18, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    # -----------------------------------
    # WARNING COUNTER
    # -----------------------------------

    cv2.putText(
        output,
        f"Warnings: {warning_count}",
        (width - 145, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if warning_count else (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return output