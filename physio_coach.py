import cv2
import mediapipe as mp
import numpy as np
import time

# ---------- MEDIAPIPE ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ---------- GLOBAL VARIABLES ----------
stage = "start"
exercise = "None"
warning = ""
stats = {
    "squat": 0,
    "curl": 0,
    "pushup": 0
}
calories = 0
CALORIES_MAP = {
    "squat": 0.32,
    "curl": 0.15,
    "pushup": 0.29
}
posture_score = 100
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

# ---------- FRAME PROCESSING ----------
def process_frame(frame):
    global stage, exercise, warning
    global stats, prev_time, posture_score, calories

    h, w, _ = frame.shape
    
    # Pre-process image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    warning = ""
    posture_score = 100

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Joint coordinates
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            ankle = [landmarks[27].x, landmarks[27].y]
            shoulder = [landmarks[11].x, landmarks[11].y]
            elbow = [landmarks[13].x, landmarks[13].y]
            wrist = [landmarks[15].x, landmarks[15].y]

            # Calculate angles
            knee_angle = calculate_angle(hip, knee, ankle)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            back_angle = calculate_angle(shoulder, hip, knee)

            # --- AUTO DETECTION ---
            if knee_angle < 140:
                exercise = "Squat"
            elif elbow_angle < 70:
                exercise = "Curl"
            elif elbow_angle < 100 and knee_angle > 150:
                exercise = "Pushup"

            # --- POSTURE LOGIC ---
            if back_angle < 150:
                posture_score -= 30
            
            # --- EXERCISE COUNTERS ---
            if exercise == "Squat":
                if knee_angle > 170: stage = "up"
                if knee_angle < 95 and stage == "up":
                    stage = "down"
                    stats["squat"] += 1
                    calories += CALORIES_MAP["squat"]
                if knee_angle > 110: warning = "GO LOWER"

            elif exercise == "Curl":
                if elbow_angle > 160: stage = "down"
                if elbow_angle < 50 and stage == "down":
                    stage = "up"
                    stats["curl"] += 1
                    calories += CALORIES_MAP["curl"]
                if elbow_angle > 90: warning = "FULL CURL"

            elif exercise == "Pushup":
                if elbow_angle > 160: stage = "up"
                if elbow_angle < 90 and stage == "up":
                    stage = "down"
                    stats["pushup"] += 1
                    calories += CALORIES_MAP["pushup"]
                hip_angle = calculate_angle(shoulder, hip, ankle)
                if hip_angle < 160: warning = "KEEP BODY STRAIGHT"

            posture_score = max(posture_score, 0)

            # Draw Landmarks
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    except Exception as e:
        pass

    # ---------- DASHBOARD UI ----------
    dashboard = np.zeros((h, 320, 3), dtype=np.uint8)
    # Create Gradient background for dashboard
    for i in range(h):
        val = 25 + int(i * 0.05)
        dashboard[i, :] = (val, val, val)

    cv2.putText(dashboard, "AI FITNESS COACH", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(dashboard, f"EXERCISE: {exercise}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(dashboard, f"Squats: {stats['squat']}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(dashboard, f"Curls: {stats['curl']}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(dashboard, f"Pushups: {stats['pushup']}", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(dashboard, f"Calories: {round(calories, 2)}", (30, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    cv2.putText(dashboard, f"Posture: {posture_score}%", (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if posture_score > 70 else (0,0,255), 2)

    # Calculate FPS
    current_time = time.time()
    fps = 1/(current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    cv2.putText(dashboard, f"FPS: {int(fps)}", (30, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Show Warning on main image
    if warning != "":
        cv2.putText(image, warning, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return np.hstack((image, dashboard))