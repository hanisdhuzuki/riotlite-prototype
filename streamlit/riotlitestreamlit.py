import streamlit as st
import cv2
import mediapipe as mp
from handtrack import opposition_exercise, stretch_exercise

# ---------- Page Setup ----------
st.set_page_config(page_title="🖐️ RIOTLite Hand Rehab", layout="wide")
st.title("🖐️ RIOTLite: Calibrated Hand Gesture Prototype")
st.write("Select an exercise and follow the on-screen feedback in real time.")

# ---------- Sidebar Controls ----------
exercise_mode = st.sidebar.selectbox(
    "Choose an exercise:", ("None", "Opposition", "Stretch")
)

if "run" not in st.session_state:
    st.session_state.run = False

start_button = st.sidebar.button("Start Camera")
stop_button = st.sidebar.button("Stop Camera")

if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False

# ---------- Video Display ----------
FRAME_WINDOW = st.image([])

# ---------- Mediapipe Setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

# ---------- Main Loop ----------
if st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Frame from the camera was not received.")
    else:
        frame = cv2.flip(frame, 1)  # mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # --- Palm width feedback ---
                index_mcp = landmarks[5]
                pinky_mcp = landmarks[17]
                width_mcp = ((index_mcp.x - pinky_mcp.x) ** 2 +
                             (index_mcp.y - pinky_mcp.y) ** 2) ** 0.5
                width_target, tolerance = 0.10, 0.03

                if width_mcp < width_target - tolerance:
                    cv2.putText(frame, 'Move your hand closer!', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif width_mcp > width_target + tolerance:
                    cv2.putText(frame, 'Move your hand further!', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # --- Exercise Execution ---
                if exercise_mode == "Opposition":
                    frame = opposition_exercise(landmarks, frame)
                elif exercise_mode == "Stretch":
                    frame = stretch_exercise(landmarks, frame)

                # --- Draw Hand Landmarks ---
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # ---------- Update Streamlit Image ----------
        FRAME_WINDOW.image(frame, channels="BGR")

# ---------- Cleanup ----------
cap.release()
