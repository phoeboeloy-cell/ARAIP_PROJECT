
from controller import Robot
import cv2 as cv
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

TIME_STEP = 32
MAX_SPEED = 6.28

# OpenCV colour detection
# Each colour uses the same minimum pixel count.
RED_PIXEL_THRESHOLD = 1000
GREEN_PIXEL_THRESHOLD = 1000
BLUE_PIXEL_THRESHOLD = 1000

# HSV ranges (OpenCV hue scale: 0-179)
LOWER_RED_1 = np.array([0, 100, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 100, 100])
UPPER_RED_2 = np.array([180, 255, 255])

LOWER_GREEN = np.array([35, 100, 100])
UPPER_GREEN = np.array([85, 255, 255])

LOWER_BLUE = np.array([90, 100, 100])
UPPER_BLUE = np.array([130, 255, 255])

# Braitenberg sensor weights
WEIGHTS = [
    [-1.3, -1.0],   # ps0
    [-1.3, -1.0],   # ps1
    [-0.5,  0.5],   # ps2
    [ 0.0,  0.0],   # ps3
    [ 0.0,  0.0],   # ps4
    [ 0.05, -0.5],  # ps5
    [-0.75, 0.0],   # ps6
    [-0.75, 0.0]    # ps7
]

OFFSETS = [
    0.5 * MAX_SPEED,
    0.5 * MAX_SPEED
]

# ==========================================
# 2. UNSTUCK CONFIGURATION
# ==========================================

STUCK_CHECK_INTERVAL = 1.0

STUCK_MOVEMENT_THRESHOLD = 0.015

STUCK_COUNT_LIMIT = 2

REVERSE_DURATION = 0.6
TURN_DURATION = 0.8

RECOVERY_SPEED = 0.5 * MAX_SPEED

# Recovery states
recovery_phase = "NORMAL"

recovery_start_time = 0.0

last_left_position = None
last_right_position = None

stuck_count = 0

turn_direction = 1

last_stuck_check = 0.0


# ==========================================
# 3. INITIALIZE ROBOT
# ==========================================

robot = Robot()

print("==========================================")
print("E-PUCK ROBOT CONTROLLER STARTED")
print("Braitenberg + OpenCV + Automatic Recovery")
print("==========================================")


# ==========================================
# 4. INITIALIZE MOTORS
# ==========================================

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

print("Motors initialized.")


# ==========================================
# 5. INITIALIZE LEDs
# ==========================================

leds = []

for i in range(10):

    led = robot.getDevice('led' + str(i))

    leds.append(led)

    led.set(0)

print("All 10 LEDs initialized.")


# ==========================================
# 6. INITIALIZE PROXIMITY SENSORS
# ==========================================

sensors = []

for i in range(8):

    sensor = robot.getDevice('ps' + str(i))

    sensor.enable(TIME_STEP)

    sensors.append(sensor)

print("All 8 proximity sensors initialized.")


# ==========================================
# 7. INITIALIZE WHEEL ENCODERS
# ==========================================

left_encoder = robot.getDevice('left wheel sensor')
right_encoder = robot.getDevice('right wheel sensor')

left_encoder.enable(TIME_STEP)
right_encoder.enable(TIME_STEP)

print("Wheel encoders initialized.")


# ==========================================
# 8. INITIALIZE CAMERA
# ==========================================

cam = robot.getDevice('camera')

cam.enable(TIME_STEP)

W = cam.getWidth()
H = cam.getHeight()

print("Camera initialized.")
print("Camera resolution:", W, "x", H)


# ==========================================
# 9. CREATE IMAGE SAVE FOLDER
# ==========================================

SAVE_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "images"
)

os.makedirs(SAVE_FOLDER, exist_ok=True)

image_count = 0

detected_before = {
    "red": False,
    "green": False,
    "blue": False
}

print("Image folder:", SAVE_FOLDER)


# ==========================================
# 10. BRAITENBERG OBSTACLE AVOIDANCE
# ==========================================

def run_braitenberg():

    left_speed = 0.0
    right_speed = 0.0

    for i in range(8):

        value = sensors[i].getValue()

        # Normalize sensor readings
        value = value / 4096.0

        left_speed += value * WEIGHTS[i][0]

        right_speed += value * WEIGHTS[i][1]

    # Apply motor offsets
    left_speed = (
        OFFSETS[0] + left_speed * MAX_SPEED
    )

    right_speed = (
        OFFSETS[1] + right_speed * MAX_SPEED
    )

    # Limit motor speeds
    left_speed = max(
        -MAX_SPEED,
        min(MAX_SPEED, left_speed)
    )

    right_speed = max(
        -MAX_SPEED,
        min(MAX_SPEED, right_speed)
    )

    return left_speed, right_speed


# ==========================================
# 11. LED BLINKING
# ==========================================

def blink_leds(step_count):

    for led in leds:
        led.set(0)

    led_index = (step_count // 10) % 10

    leds[led_index].set(1)


# ==========================================
# 12. CONVERT WEBOTS IMAGE TO OPENCV
# ==========================================

def get_camera_frame():

    img = cam.getImage()

    if img is None:
        return None

    image_array = np.frombuffer(
        img,
        dtype=np.uint8
    )

    image_array = image_array.reshape(
        (H, W, 4)
    )

    # Convert BGRA to BGR
    image_bgr = cv.cvtColor(
        image_array,
        cv.COLOR_BGRA2BGR
    )

    return image_bgr


# ==========================================
# ==========================================
# 13. OPENCV RED, GREEN AND BLUE DETECTION
# ==========================================

def detect_red_object(image):

    if image is None:
        return False, 0

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Red needs two HSV ranges because it wraps around hue 0.
    mask1 = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv.bitwise_or(mask1, mask2)

    red_pixels = cv.countNonZero(red_mask)
    detected = red_pixels >= RED_PIXEL_THRESHOLD

    return detected, red_pixels


def detect_green_object(image):

    if image is None:
        return False, 0

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    green_mask = cv.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    green_pixels = cv.countNonZero(green_mask)
    detected = green_pixels >= GREEN_PIXEL_THRESHOLD

    return detected, green_pixels


def detect_blue_object(image):

    if image is None:
        return False, 0

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    blue_mask = cv.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    blue_pixels = cv.countNonZero(blue_mask)
    detected = blue_pixels >= BLUE_PIXEL_THRESHOLD

    return detected, blue_pixels


# 14. SAVE CAMERA IMAGE
# ==========================================

def save_camera_image(image, colour):

    global image_count

    if image is None:

        print("Camera image unavailable.")

        return

    image_count += 1

    filename = (
        f"{colour}_object_{image_count:03d}.jpg"
    )

    filepath = os.path.join(
        SAVE_FOLDER,
        filename
    )

    success = cv.imwrite(
        filepath,
        image
    )

    if success:

        print("Image successfully saved!")
        print("Filename:", filename)
        print("Location:", filepath)

    else:

        print("Failed to save image.")


# ==========================================
# 15. DISPLAY SENSOR INFORMATION
# ==========================================

def display_sensor_information(
    step_count,
    left_speed,
    right_speed
):

    readings = [
        round(sensor.getValue(), 1)
        for sensor in sensors
    ]

    print("\nStep:", step_count)

    print("Proximity sensor readings:", readings)

    print(
        f"Motor speeds | "
        f"Left: {left_speed:.2f} | "
        f"Right: {right_speed:.2f}"
    )


# ==========================================
# 16. STOP ROBOT
# ==========================================

def stop_robot():

    left_motor.setVelocity(0.0)

    right_motor.setVelocity(0.0)


# ==========================================
# 17. RESET STUCK MONITOR
# ==========================================

def reset_stuck_monitor():

    global last_left_position
    global last_right_position
    global last_stuck_check
    global stuck_count
    global recovery_phase

    # Reset recovery after colour detection
    recovery_phase = "NORMAL"

    stuck_count = 0

    # Update encoder baseline
    last_left_position = left_encoder.getValue()

    last_right_position = right_encoder.getValue()

    last_stuck_check = robot.getTime()


# ==========================================
# 18. AUTOMATIC UNSTUCK SYSTEM
# ==========================================

def unstuck_robot(left_speed, right_speed):

    global recovery_phase
    global recovery_start_time
    global last_left_position
    global last_right_position
    global stuck_count
    global turn_direction
    global last_stuck_check

    current_time = robot.getTime()

    # Read wheel encoders
    left_position = left_encoder.getValue()

    right_position = right_encoder.getValue()

    # --------------------------------------
    # A. REVERSE PHASE
    # --------------------------------------

    if recovery_phase == "REVERSE":

        if (
            current_time - recovery_start_time
            < REVERSE_DURATION
        ):

            left_motor.setVelocity(-RECOVERY_SPEED)

            right_motor.setVelocity(-RECOVERY_SPEED)

            return True

        else:

            recovery_phase = "TURN"

            recovery_start_time = current_time

            print("Recovery: Turning away from obstacle.")

    # --------------------------------------
    # B. TURN PHASE
    # --------------------------------------

    if recovery_phase == "TURN":

        if (
            current_time - recovery_start_time
            < TURN_DURATION
        ):

            left_motor.setVelocity(
                turn_direction * RECOVERY_SPEED
            )

            right_motor.setVelocity(
                -turn_direction * RECOVERY_SPEED
            )

            return True

        else:

            recovery_phase = "NORMAL"

            stuck_count = 0

            last_left_position = left_position

            last_right_position = right_position

            last_stuck_check = current_time

            print("Recovery completed!")

            print("Resuming Braitenberg navigation.")

            return False

    # --------------------------------------
    # C. INITIALIZE ENCODER BASELINE
    # --------------------------------------

    if last_left_position is None:

        last_left_position = left_position

        last_right_position = right_position

        last_stuck_check = current_time

        return False

    # --------------------------------------
    # D. CHECK ROBOT MOVEMENT
    # --------------------------------------

    if (
        current_time - last_stuck_check
        >= STUCK_CHECK_INTERVAL
    ):

        left_change = abs(
            left_position - last_left_position
        )

        right_change = abs(
            right_position - last_right_position
        )

        total_movement = (
            left_change + right_change
        )

        # Check whether motors are commanded to move
        motor_active = (
            abs(left_speed) > 1.0 or
            abs(right_speed) > 1.0
        )

        if (
            motor_active and
            total_movement < STUCK_MOVEMENT_THRESHOLD
        ):

            stuck_count += 1

            print(
                "Possible stuck condition:",
                stuck_count
            )

        else:

            stuck_count = 0

        # Update encoder positions
        last_left_position = left_position

        last_right_position = right_position

        last_stuck_check = current_time

        # ----------------------------------
        # E. ACTIVATE RECOVERY
        # ----------------------------------

        if stuck_count >= STUCK_COUNT_LIMIT:

            print("\n================================")

            print("ROBOT IS STUCK!")

            print("Starting automatic recovery.")

            print("================================\n")

            # Compare left and right obstacles
            left_obstacle = (
                sensors[6].getValue() +
                sensors[7].getValue()
            )

            right_obstacle = (
                sensors[0].getValue() +
                sensors[1].getValue()
            )

            # Choose direction with fewer obstacles
            if left_obstacle > right_obstacle:

                turn_direction = -1

                print("Recovery direction: RIGHT")

            else:

                turn_direction = 1

                print("Recovery direction: LEFT")

            # Start reverse phase
            recovery_phase = "REVERSE"

            recovery_start_time = current_time

            stuck_count = 0

            return True

    return False


# ==========================================
# 19. MAIN PROGRAM
# ==========================================

def main():

    step_count = 0

    print("\nE-puck search operation started!")

    print("Obstacle avoidance: ACTIVE")

    print("LED animation: ACTIVE")

    print("OpenCV red, green and blue detection: ACTIVE")

    print("Automatic unstuck system: ACTIVE")

    print("------------------------------------------")

    while robot.step(TIME_STEP) != -1:

        step_count += 1

        # ----------------------------------
        # A. LED BLINKING
        # ----------------------------------

        blink_leds(step_count)

        # ----------------------------------
        # B. CAMERA IMAGE PROCESSING
        # ----------------------------------

        camera_image = get_camera_frame()

        red_detected, red_pixels = detect_red_object(camera_image)
        green_detected, green_pixels = detect_green_object(camera_image)
        blue_detected, blue_pixels = detect_blue_object(camera_image)

        # Collect colours that meet their pixel threshold.
        detections = []

        if red_detected:
            detections.append(("red", red_pixels))

        if green_detected:
            detections.append(("green", green_pixels))

        if blue_detected:
            detections.append(("blue", blue_pixels))

        # ----------------------------------
        # C. COLOUR DETECTION PRIORITY
        # ----------------------------------

        if detections:

            # If multiple colours are detected, handle the one
            # with the largest detected pixel area.
            detected_colour, detected_pixels = max(
                detections,
                key=lambda item: item[1]
            )

            # Stop robot immediately for any target colour.
            stop_robot()
            reset_stuck_monitor()

            if not detected_before[detected_colour]:

                print("\\n================================")
                print(f"{detected_colour.upper()} OBJECT DETECTED!")
                print(
                    f"{detected_colour.capitalize()} pixels detected:",
                    detected_pixels
                )
                print("Robot stopping.")

                # Save image using the detected colour in its filename.
                save_camera_image(camera_image, detected_colour)

                print("================================\\n")

            # Track which colours remain visible.
            for colour in detected_before:
                detected_before[colour] = any(
                    item[0] == colour for item in detections
                )

            continue

        # Reset detection flags when no target colour is visible.
        for colour in detected_before:
            detected_before[colour] = False

        # D. BRAITENBERG OBSTACLE AVOIDANCE
        # ----------------------------------

        left_speed, right_speed = run_braitenberg()

        # ----------------------------------
        # E. AUTOMATIC UNSTUCK SYSTEM
        # ----------------------------------

        is_recovering = unstuck_robot(
            left_speed,
            right_speed
        )

        # Apply normal movement only if not recovering
        if not is_recovering:

            left_motor.setVelocity(left_speed)

            right_motor.setVelocity(right_speed)

        # ----------------------------------
        # F. DISPLAY SENSOR INFORMATION
        # ----------------------------------

        if step_count % 32 == 0:

            display_sensor_information(
                step_count,
                left_speed,
                right_speed
            )


# ==========================================
# 20. RUN PROGRAM
# ==========================================

if __name__ == '__main__':

    main()