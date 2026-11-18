
from controller import Robot
import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================

TIME_STEP = 32
MAX_SPEED = 6.28

# Create camera image folder
SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "camera_images"
)

os.makedirs(SAVE_PATH, exist_ok=True)

# Default e-puck Braitenberg sensor weights
WEIGHTS = [
    [-1.3, -1.0],
    [-1.3, -1.0],
    [-0.5,  0.5],
    [ 0.0,  0.0],
    [ 0.0,  0.0],
    [ 0.05, -0.5],
    [-0.75, 0.0],
    [-0.75, 0.0]
]

OFFSETS = [0.5 * MAX_SPEED, 0.5 * MAX_SPEED]

COLOUR_THRESHOLD = 0.15


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    robot = Robot()

    print("PYTHON CONTROLLER STARTED!")
    print("OpenCV colour detection initialized.")
    print("Camera image saving initialized.")
    print("Images will be saved in:", SAVE_PATH)

    # ======================================
    # 1. INITIALIZE MOTORS
    # ======================================

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')

    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ======================================
    # 2. INITIALIZE LEDS
    # ======================================

    leds = []

    for i in range(10):
        led = robot.getDevice('led' + str(i))
        leds.append(led)
        led.set(0)

    print("All LEDs initialized.")

    # ======================================
    # 3. INITIALIZE PROXIMITY SENSORS
    # ======================================

    sensors = []

    for i in range(8):
        sensor = robot.getDevice('ps' + str(i))
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    print("All 8 proximity sensors initialized!")

    # ======================================
    # 4. INITIALIZE CAMERA
    # ======================================

    cam = robot.getDevice('camera')
    cam.enable(TIME_STEP)

    W = cam.getWidth()
    H = cam.getHeight()

    print("Camera initialized.")
    print("Camera resolution:", W, "x", H)

    # ======================================
    # 5. TRACK ENCOUNTERED COLOURS
    # ======================================

    encountered = {
        "red": False,
        "green": False,
        "blue": False
    }

    last_detected_color = None

    # ======================================
    # 6. BRAITENBERG OBSTACLE AVOIDANCE
    # ======================================

    def run_braitenberg():

        left_speed = 0.0
        right_speed = 0.0

        for i in range(8):

            value = sensors[i].getValue()
            value = value / 4096.0

            left_speed += value * WEIGHTS[i][0]
            right_speed += value * WEIGHTS[i][1]

        left_speed = OFFSETS[0] + left_speed * MAX_SPEED
        right_speed = OFFSETS[1] + right_speed * MAX_SPEED

        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        return left_speed, right_speed

    # ======================================
    # 7. LED BLINKING
    # ======================================

    def blink_leds(step_count):

        for led in leds:
            led.set(0)

        led_index = (step_count // 10) % 10
        leds[led_index].set(1)

    # ======================================
    # 8. OPENCV COLOUR DETECTION
    # ======================================

    def detect_colour(step_count):

        img = cam.getImage()

        if img is None:
            return None

        # Convert Webots camera image into NumPy array
        image = np.frombuffer(img, dtype=np.uint8)
        image = image.reshape((H, W, 4))

        # Convert BGRA into BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # ==================================
        # SAVE CAMERA IMAGE
        # ==================================

        if step_count % 32 == 0:

            filename = os.path.join(
                SAVE_PATH,
                f"camera_{step_count}.png"
            )

            cv2.imwrite(filename, bgr)

            print("Camera image saved:", filename)

        # Convert BGR into HSV
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Select central camera region
        x1 = int(W * 0.35)
        x2 = int(W * 0.65)

        y1 = int(H * 0.35)
        y2 = int(H * 0.65)

        roi = hsv[y1:y2, x1:x2]

        # ==================================
        # RED DETECTION
        # ==================================

        red_lower1 = np.array([0, 100, 80])
        red_upper1 = np.array([10, 255, 255])

        red_lower2 = np.array([170, 100, 80])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(roi, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(roi, red_lower2, red_upper2)

        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # ==================================
        # GREEN DETECTION
        # ==================================

        green_lower = np.array([35, 80, 80])
        green_upper = np.array([85, 255, 255])

        green_mask = cv2.inRange(
            roi, green_lower, green_upper
        )

        # ==================================
        # BLUE DETECTION
        # ==================================

        blue_lower = np.array([90, 80, 80])
        blue_upper = np.array([135, 255, 255])

        blue_mask = cv2.inRange(
            roi, blue_lower, blue_upper
        )

        # ==================================
        # SAVE COLOUR DETECTION MASKS
        # ==================================

        if step_count % 32 == 0:

            cv2.imwrite(
                os.path.join(SAVE_PATH, f"red_{step_count}.png"),
                red_mask
            )

            cv2.imwrite(
                os.path.join(SAVE_PATH, f"green_{step_count}.png"),
                green_mask
            )

            cv2.imwrite(
                os.path.join(SAVE_PATH, f"blue_{step_count}.png"),
                blue_mask
            )

        # ==================================
        # CALCULATE COLOUR PIXEL PERCENTAGE
        # ==================================

        total_pixels = roi.shape[0] * roi.shape[1]

        red_percentage = cv2.countNonZero(red_mask) / total_pixels
        green_percentage = cv2.countNonZero(green_mask) / total_pixels
        blue_percentage = cv2.countNonZero(blue_mask) / total_pixels

        colour_percentages = {
            "red": red_percentage,
            "green": green_percentage,
            "blue": blue_percentage
        }

        detected = max(
            colour_percentages,
            key=colour_percentages.get
        )

        if colour_percentages[detected] >= COLOUR_THRESHOLD:
            return detected

        return None

    # ======================================
    # 9. START OPERATION
    # ======================================

    print("E-puck search operation started!")
    print("OpenCV is detecting red, green and blue.")

    step_count = 0

    # ======================================
    # 10. MAIN CONTROL LOOP
    # ======================================

    while robot.step(TIME_STEP) != -1:

        step_count += 1

        # OBSTACLE AVOIDANCE
        left_speed, right_speed = run_braitenberg()

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # LED BLINKING
        blink_leds(step_count)

        # DISPLAY SENSOR INFORMATION
        if step_count % 32 == 0:

            readings = [
                round(sensor.getValue(), 1)
                for sensor in sensors
            ]

            print("Proximity sensors:", readings)

            print(
                f"Motor speeds | "
                f"Left: {left_speed:.2f} | "
                f"Right: {right_speed:.2f}"
            )

        # OPENCV COLOUR DETECTION
        detected_color = detect_colour(step_count)

        # UNIQUE COLOUR TRACKING
        if detected_color is not None:

            if detected_color != last_detected_color:

                encountered[detected_color] = True
                last_detected_color = detected_color

                print(f"OpenCV detected {detected_color}!")

                active_items = [
                    color.capitalize()
                    for color, found in encountered.items()
                    if found
                ]

                print(
                    "Unique colours encountered so far: "
                    + ', '.join(active_items)
                )

                print("-" * 40)

        else:
            last_detected_color = None


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == '__main__':
    main()
