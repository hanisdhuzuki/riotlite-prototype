import cv2
import mediapipe as mp
import csv
from datetime import datetime


# <==functions: Exercises ==>.

# 1) Opposition Exercise.
def opposition_exercise(landmarks, frame):
    hand_marks = [8, 12, 16, 20]  # fingertips
    y = 100
    max_dist = 0.3
    percentages = []

    for i in hand_marks:
        finger_tip = landmarks[i]
        thumb = landmarks[4]
        distance = ((finger_tip.x - thumb.x) ** 2 + (finger_tip.y - thumb.y) ** 2) ** 0.5
        percent_num = max(0, min(1, 1 - distance / max_dist)) * 100
        percentages.append(percent_num)

    for i, percent in enumerate(percentages):
        cv2.putText(frame, f'Sentuh ibu jari dgn jari {i+1}: {percent:.2f}%',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y += 30
    return frame, percentages
    #return frame


# 2) Stretch Exercise.
def stretch_exercise(landmarks, frame):
    hand_marks = [8, 12, 16, 20]  # fingertips
    y = 100
    max_dist = 0.3
    wrist = landmarks[0]
    stretch_percentages = []

    for i in hand_marks:
        fingertip = landmarks[i]
        distance = ((fingertip.x - wrist.x) ** 2 + (fingertip.y - wrist.y) ** 2) ** 0.5
        percent_num = max(0, min(1, distance / max_dist)) * 100
        stretch_percentages.append(percent_num)

    avg_stretch = sum(stretch_percentages) / len(stretch_percentages)
    cv2.putText(frame, f'Purata %: {avg_stretch:.2f}%',
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if avg_stretch > 15:
        cv2.putText(frame, 'Buka genggaman tangan anda',
                    (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'Cuba buka lagi',
                    (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, [avg_stretch]
    #return frame

# <---------------Setup--------------->

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

exercise_mode = None  # Default no exercise.
exercises = {
    "opposition": opposition_exercise,
    "stretch": stretch_exercise
}

# CSV logging (kiv)
log_file = "riotlite_log.csv"

#<---------------Main Loop--------------->

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot receive frame")
        break

    # Flip for selfie-view
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Estimate palm width → distance from camera
            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]
            width_mcp = ((index_mcp.x - pinky_mcp.x) ** 2 + (index_mcp.y - pinky_mcp.y) ** 2) ** 0.5

            width_target = 0.10  # ~30 cm
            tolerance = 0.03

            if width_mcp < width_target - tolerance:
                cv2.putText(frame, 'Dekatkan tangan dengan kamera',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif width_mcp > width_target + tolerance:
                cv2.putText(frame, 'Jauhkan tangan dari kamera',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Run selected exercise
            if exercise_mode in exercises:
                frame, values = exercises[exercise_mode](landmarks, frame)

                # Optional logging
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now(), exercise_mode, values])

            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    if exercise_mode:
        cv2.putText(frame, f'Mode: {exercise_mode.capitalize()}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, 'Tekan 1: Opposition | 2: Stretch',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    
#     cv2.imshow('RIOTLite Motor Assessment Scale: Hand Movements', frame)

#     # Key controls
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('1'):
#         exercise_mode = "opposition"
#     elif key == ord('2'):
#         exercise_mode = "stretch"
#     elif key == ord('q') or key == 27:  # q or ESC
#         break

# # Release
# cap.release()
# cv2.destroyAllWindows()

