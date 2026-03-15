# updates
import cv2
import mediapipe as mp
import numpy as np
import time

# ---------- MEDIAPIPE SETUP ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils


stage = "start"
exercise = "None"
warning = ""

stats = {
    "squat": 0,
    "curl": 0,
    "pushup": 0
}

prev_time = 0


# ---------- ANGLE FUNCTION ----------
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# ---------- MAIN FRAME PROCESSING FUNCTION ----------
def process_frame(frame):

    global stage, exercise, warning, stats, prev_time

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    warning = ""

    try:

        landmarks = results.pose_landmarks.landmark

        # joints
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]

        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]

        # angles
        knee_angle = calculate_angle(hip, knee, ankle)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        back_angle = calculate_angle(shoulder, hip, knee)

        # ---------- AUTO EXERCISE DETECTION ----------
        if knee_angle < 140:
            exercise = "Squat"

        elif elbow_angle < 70:
            exercise = "Curl"

        elif elbow_angle < 100 and knee_angle > 150:
            exercise = "Pushup"

        # ---------- SQUAT ----------
        if exercise == "Squat":

            if knee_angle > 170:
                stage = "up"

            if knee_angle < 95 and stage == "up":
                stage = "down"
                stats["squat"] += 1

            if knee_angle > 110:
                warning = "GO LOWER"

            if back_angle < 150:
                warning = "KEEP BACK STRAIGHT"

        # ---------- CURL ----------
        elif exercise == "Curl":

            if elbow_angle > 160:
                stage = "down"

            if elbow_angle < 50 and stage == "down":
                stage = "up"
                stats["curl"] += 1

            if elbow_angle > 90:
                warning = "FULL CURL"

        # ---------- PUSHUP ----------
        elif exercise == "Pushup":

            if elbow_angle > 160:
                stage = "up"

            if elbow_angle < 90 and stage == "up":
                stage = "down"
                stats["pushup"] += 1

            hip_angle = calculate_angle(shoulder, hip, ankle)

            if hip_angle < 160:
                warning = "KEEP BODY STRAIGHT"

    except:
        pass

    # ---------- DASHBOARD ----------
    dashboard = np.zeros((h, 320, 3), dtype=np.uint8)

    for i in range(h):
        c = 25 + int(i * 0.05)
        dashboard[i, :] = (c, c, c)

    cv2.putText(dashboard, "AI FITNESS COACH", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)

    cv2.putText(dashboard, "EXERCISE", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.putText(dashboard, exercise, (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(dashboard, "WORKOUT STATS", (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(dashboard, f"Squats: {stats['squat']}", (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    cv2.putText(dashboard, f"Curls: {stats['curl']}", (30, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    cv2.putText(dashboard, f"Pushups: {stats['pushup']}", (30, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    # ---------- WARNING ----------
    if warning != "":
        cv2.putText(image,
                    warning,
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # ---------- FPS ----------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(dashboard, f"FPS {int(fps)}", (30, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ---------- DRAW POSE ----------
    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

    combined = np.hstack((image, dashboard))

    return combined