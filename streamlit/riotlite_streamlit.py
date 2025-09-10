import cv2
import streamlit as st
import mediapipe as mp
from handtrack import opposition_exercise, stretch_exercise

st.title("🖐️ RIOTLite: A Calibrated Hand Gesture Prototype")
st.write("**Motor Assessment Scale (Hand Movements) / Skala Penilaian Motor (Pergerakan Tangan)**")

if "cap" not in st.session_state:
    st.session_state.cap = None
if "running" not in st.session_state:
    st.session_state.running = False

start = st.button("▶ Buka Kamera / Start Camera")
stop = st.button("⏹ Tutup Kamera / Stop Camera")
mode = st.selectbox("Pilih Mod Latihan / Select Exercise Mode", ("None", "Opposition", "Stretch"))
st.session_state.mode = mode

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FRAME_WINDOW = st.empty()

if start and not st.session_state.running:
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.running = True
    st.write("Camera started")

if stop and st.session_state.running:
    st.session_state.cap.release()
    st.session_state.cap = None
    st.session_state.running = False

if st.session_state.running and st.session_state.cap is not None:
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    ret, frame = st.session_state.cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                if st.session_state.mode == "Opposition":
                    frame, _ = opposition_exercise(landmarks, frame)
                elif st.session_state.mode == "Stretch":
                    frame, _ = stretch_exercise(landmarks, frame)

        rgb_display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_display_frame, channels="RGB")
    else:
        st.warning("⚠️ Kamera tidak disambung / Camera not available")
        st.session_state.cap.release()
        st.session_state.cap = None
        st.session_state.running = False
