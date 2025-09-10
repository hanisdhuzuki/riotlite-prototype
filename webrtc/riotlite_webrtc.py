import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from handtrack import opposition_exercise, stretch_exercise

# ---------- Page Setup ----------
st.set_page_config(page_title="🖐️ RIOTLite Hand Rehab", layout="wide")
st.title("🖐️ RIOTLite: Calibrated Hand Gesture Prototype")
st.write("Select an exercise and follow the on-screen feedback in real time.")

# ---------- Sidebar Controls ----------
exercise_mode = st.sidebar.selectbox(
    "Choose an exercise:", ("None", "Opposition", "Stretch")
)

# ---------- Mediapipe Setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # --- Palm width feedback ---
                index_mcp = landmarks[5]
                pinky_mcp = landmarks[17]
                width_mcp = ((index_mcp.x - pinky_mcp.x) ** 2 + (index_mcp.y - pinky_mcp.y) ** 2) ** 0.5
                width_target = 0.10
                tolerance = 0.03

                if width_mcp < width_target - tolerance:
                    cv2.putText(img, 'Move your hand closer!', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif width_mcp > width_target + tolerance:
                    cv2.putText(img, 'Move your hand further!', (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # --- Exercise Execution ---
                if exercise_mode == "Opposition":
                    img = opposition_exercise(landmarks, img)
                elif exercise_mode == "Stretch":
                    img = stretch_exercise(landmarks, img)

                # --- Draw Hand Landmarks ---
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- Start WebRTC Stream ----------
webrtc_streamer(key="riotlite", video_processor_factory=VideoProcessor)
