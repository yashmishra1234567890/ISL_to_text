
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os
from collections import deque, Counter


# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
model = load_model("best_lstm_sign_model_60.h5")

# Mediapipe setup
mp_holistic = mp.solutions.holistic

def extract_keypoints(results, frame_shape):
    def norm_landmarks(landmarks, frame_shape, with_visibility=False):
        if not landmarks:
            return []
        w, h = frame_shape[1], frame_shape[0]
        arr = []
        for lm in landmarks:
            arr.extend([lm.x * w, lm.y * h, lm.z * w])
            if with_visibility:
                arr.append(lm.visibility)
        return arr
    pose = norm_landmarks(results.pose_landmarks.landmark if results.pose_landmarks else [], frame_shape, with_visibility=True)
    if not pose:
        pose = [0.0] * (33 * 4)
    lh = norm_landmarks(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], frame_shape)
    if not lh:
        lh = [0.0] * (21 * 3)
    rh = norm_landmarks(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], frame_shape)
    if not rh:
        rh = [0.0] * (21 * 3)
    keypoints = np.array(pose + lh + rh, dtype=np.float32)
    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
    if len(keypoints) < 322:
        keypoints = np.pad(keypoints, (0, 322 - len(keypoints)), mode='constant', constant_values=0.0)
    elif len(keypoints) > 322:
        keypoints = keypoints[:322]
    return keypoints.astype(np.float32)

sequence_length = 30
sequence = deque(maxlen=sequence_length)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        keypoints = extract_keypoints(results, frame.shape)
        sequence.append(keypoints)
        if len(sequence) == sequence_length:
            input_data = np.expand_dims(sequence, axis=0)
            predictions = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions)
            label = labels[predicted_index]
            cv2.putText(frame, f"{label} ({confidence:.2f})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
