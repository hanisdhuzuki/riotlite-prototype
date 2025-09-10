import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
from handtrack import opposition_exercise, stretch_exercise

# ---------- Page Setup ----------
st.set_page_config(page_title="🖐️ RIOTLite WebRTC Hand Rehab", layout="wide")
st.title("🖐️ RIOTLite: Calibrated Hand Gesture Prototype (WebRTC)")
st.write("Live hand rehab exercises via your browser camera.")

# ---------- Sidebar Controls ----------
exercise_mode = st.sidebar.selectbox(
    "Select an exercise:", ("None", "Opposition", "Stretch")
)

st.sidebar.markdown("---")
st.sidebar.write("✅ Ensure your camera is connected and allowed in the browser.")

# ---------- Mediapipe Setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---------- Video Processing Callback ----------
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")  # convert frame to OpenCV image
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # --- Exercise Feedback ---
            if exercise_mode == "Opposition":
                img = opposition_exercise(landmarks, img)
            elif exercise_mode == "Stretch":
                img = stretch_exercise(landmarks, img)

            # --- Draw Hand Landmarks ---
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return img

# ---------- Start WebRTC Stream ----------
webrtc_streamer(
    key="hand-rehab",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
