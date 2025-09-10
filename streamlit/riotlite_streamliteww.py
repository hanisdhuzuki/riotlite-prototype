import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# ---------- Exercises Functions ----------
def opposition_exercise(landmarks, frame):
    hand_marks = [8, 12, 16, 20]  # fingertips
    y = 30
    max_dist = 0.3
    percentages = []

    for i in hand_marks:
        finger_tip = landmarks[i]
        thumb_tip = landmarks[4]
        dist = ((finger_tip.x - thumb_tip.x)**2 + (finger_tip.y - thumb_tip.y)**2)**0.5
        percent = min(dist / max_dist, 1.0) * 100
        percentages.append(percent)
        cv2.putText(frame, f'{percent:.0f}%', 
                    (int(finger_tip.x*frame.shape[1]), int(finger_tip.y*frame.shape[0])-y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y += 20
    return percentages, frame

def stretch_exercise(landmarks, frame):
    hand_marks = [8, 12, 16, 20]
    percentages = []
    y = 30
    max_dist = 0.3

    for i in hand_marks:
        finger_tip = landmarks[i]
        wrist = landmarks[0]
        dist = ((finger_tip.x - wrist.x)**2 + (finger_tip.y - wrist.y)**2)**0.5
        percent = min(dist / max_dist, 1.0) * 100
        percentages.append(percent)
        cv2.putText(frame, f'{percent:.0f}%', 
                    (int(finger_tip.x*frame.shape[1]), int(finger_tip.y*frame.shape[0])-y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        y += 20
    return percentages, frame

# ---------- Streamlit UI ----------
st.title("🖐️ RIOTLite: Hand Gesture Rehab Prototype")
st.write("Camera feed below. Press 'Stop' to end session.")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])  # placeholder for video frames

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = handLms.landmark
            opp_percent, frame = opposition_exercise(landmarks, frame)
            stretch_percent, frame = stretch_exercise(landmarks, frame)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
