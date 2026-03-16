import cv2
import mediapipe as mp
import numpy as np
import time

# ---------- MEDIAPIPE ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils


# ---------- VARIABLES ----------
stage = "start"
exercise = "None"
warning = ""

stats = {
    "squat":0,
    "curl":0,
    "pushup":0
}

# ---------- CALORIES ----------
calories = 0

CALORIES_MAP = {
    "squat":0.32,
    "curl":0.15,
    "pushup":0.29
}

posture_score = 100
prev_time = 0


# ---------- ANGLE FUNCTION ----------
def calculate_angle(a,b,c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


# ---------- FRAME PROCESSING ----------
def process_frame(frame):

    global stage, exercise, warning
    global stats, prev_time, posture_score, calories

    frame = cv2.flip(frame,1)

    h,w,_ = frame.shape

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    warning = ""
    posture_score = 100

    try:

        if not results.pose_landmarks:
            return frame

        landmarks = results.pose_landmarks.landmark

        # joints
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]

        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]

        # angles
        knee_angle = calculate_angle(hip,knee,ankle)
        elbow_angle = calculate_angle(shoulder,elbow,wrist)
        back_angle = calculate_angle(shoulder,hip,knee)

        # ---------- AUTO EXERCISE DETECTION ----------
        if knee_angle < 140:
            exercise = "Squat"

        elif elbow_angle < 70:
            exercise = "Curl"

        elif elbow_angle < 100 and knee_angle > 150:
            exercise = "Pushup"


        # ---------- POSTURE SCORE ----------
        if back_angle < 150:
            posture_score -= 30

        if exercise == "Squat" and knee_angle > 120:
            posture_score -= 20

        if exercise == "Curl" and elbow_angle > 90:
            posture_score -= 20

        posture_score = max(posture_score,0)


        # ---------- SQUAT ----------
        if exercise == "Squat":

            if knee_angle > 170:
                stage = "up"

            if knee_angle < 95 and stage == "up":
                stage = "down"
                stats["squat"] += 1
                calories += CALORIES_MAP["squat"]

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
                calories += CALORIES_MAP["curl"]

            if elbow_angle > 90:
                warning = "FULL CURL"


        # ---------- PUSHUP ----------
        elif exercise == "Pushup":

            if elbow_angle > 160:
                stage = "up"

            if elbow_angle < 90 and stage == "up":
                stage = "down"
                stats["pushup"] += 1
                calories += CALORIES_MAP["pushup"]

            hip_angle = calculate_angle(shoulder,hip,ankle)

            if hip_angle < 160:
                warning = "KEEP BODY STRAIGHT"

    except:
        pass


    # ---------- DASHBOARD ----------
    dashboard = np.zeros((h,320,3),dtype=np.uint8)

    for i in range(h):
        c = 25 + int(i*0.05)
        dashboard[i,:] = (c,c,c)

    cv2.putText(dashboard,"AI FITNESS COACH",(30,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,200),2)

    cv2.putText(dashboard,"EXERCISE",(30,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)

    cv2.putText(dashboard,exercise,(30,170),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    cv2.putText(dashboard,f"Squats: {stats['squat']}",(30,260),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)

    cv2.putText(dashboard,f"Curls: {stats['curl']}",(30,300),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)

    cv2.putText(dashboard,f"Pushups: {stats['pushup']}",(30,340),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)

    cv2.putText(dashboard,f"Calories: {round(calories,2)} kcal",
                (30,380),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,180,0),2)

    cv2.putText(dashboard,f"Posture: {posture_score}%",
                (30,420),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    current_time = time.time()
    fps = 1/(current_time-prev_time) if prev_time!=0 else 0
    prev_time = current_time

    cv2.putText(dashboard,f"FPS {int(fps)}",(30,460),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    combined = np.hstack((image,dashboard))

    return combined