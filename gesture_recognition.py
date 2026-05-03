"""
╔══════════════════════════════════════════════════════════════╗
║   GESTURE RECOGNITION - PHASE 2                             ║
║   TikTok Controller + Incremental Learning                  ║
╚══════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install opencv-python mediapipe numpy pyautogui
    pip install scikit-learn pandas

Run:
    python gesture_recognition.py

Controls:
    C     = Correction mode (incremental calibration)
    Q     = Quit
    S     = Screenshot
"""

import cv2  # pyre-ignore[21]
import mediapipe as mp  # pyre-ignore[21]
import numpy as np  # pyre-ignore[21]
import time
import pyautogui  # pyre-ignore[21]
from collections import deque

pyautogui.FAILSAFE = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "camera_index": 0,          # Ganti ke 1 jika webcam eksternal
    "frame_width": 1280,
    "frame_height": 720,
    "detection_confidence": 0.75,
    "tracking_confidence": 0.7,
    "smoothing_window": 15,     # Naikkan buffer — butuh lebih banyak frame konsisten
    "gesture_hold_frames": 10,  # Gesture harus konsisten 10 frame sebelum dikonfirmasi
}

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─────────────────────────────────────────────
# LANDMARK INDEX REFERENCE
# ─────────────────────────────────────────────
# MediaPipe 21 landmarks:
#   0: WRIST
#   1-4: THUMB (CMC, MCP, IP, TIP)
#   5-8: INDEX FINGER (MCP, PIP, DIP, TIP)
#   9-12: MIDDLE FINGER (MCP, PIP, DIP, TIP)
#   13-16: RING FINGER (MCP, PIP, DIP, TIP)
#   17-20: PINKY (MCP, PIP, DIP, TIP)



import pickle
import os

# ─────────────────────────────────────────────
# ML MODEL LOADER
# ─────────────────────────────────────────────

MODEL_FILE = "gesture_model.pkl"
_ml_bundle = None

def load_model():
    global _ml_bundle
    if not os.path.exists(MODEL_FILE):
        print(f"[WARNING] {MODEL_FILE} tidak ditemukan, pakai rule-based fallback")
        return False
    with open(MODEL_FILE, "rb") as f:
        _ml_bundle = pickle.load(f)
    print(f"[INFO] ML model loaded — labels: {_ml_bundle['labels']}")
    return True

def extract_features(landmarks):
    """42 fitur: (x,y) 21 landmark, dinormalisasi relatif wrist."""
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    features = []
    for lm in landmarks:
        features.append(lm.x - wrist_x)
        features.append(lm.y - wrist_y)
    scale = max(
        abs(landmarks[9].x - wrist_x) + abs(landmarks[9].y - wrist_y),
        1e-6
    )
    return [f / scale for f in features]

def classify_gesture(landmarks):
    """
    ML-based gesture classifier.
    Returns: (gesture_name, confidence_level)
    """
    if _ml_bundle is None:
        return ("UNKNOWN", "LOW")

    features = extract_features(landmarks)
    X = _ml_bundle["scaler"].transform([features])
    model = _ml_bundle["model"]

    prediction = model.predict(X)[0]

    # Ambil confidence dari jarak ke tetangga terdekat
    distances, _ = model.kneighbors(X)
    avg_dist = distances[0].mean()
    if avg_dist < 0.3:
        confidence = "HIGH"
    elif avg_dist < 0.7:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Map label CSV ke label display
    label_map = {
        "SCROLL_UP":   "SCROLL UP",
        "SCROLL_DOWN": "SCROLL DOWN",
        "LIKE":        "LIKE",
        "STOP":        "STOP",
        "UNKNOWN":     "UNKNOWN",
    }
    return (label_map.get(prediction, prediction), confidence)





# ─────────────────────────────────────────────
# GESTURE SMOOTHER (Anti-flickering)
# ─────────────────────────────────────────────

class GestureSmoother:
    """
    Stabilisasi hasil gesture dengan majority voting dari buffer frame terakhir.
    Mencegah gesture berkedip/flickering.
    """
    def __init__(self, window=8, hold=5):
        self.buffer = deque(maxlen=window)
        self.hold = hold
        self.confirmed = "UNKNOWN"
        self.hold_counter = 0

    def update(self, gesture):
        self.buffer.append(gesture)
        if len(self.buffer) < self.hold:
            return self.confirmed

        # Majority voting
        counts: dict[str, int] = {}
        for g in self.buffer:
            counts[g] = counts.get(g, 0) + 1
        majority = max(counts, key=lambda k: counts[k])
        majority_count = counts[majority]

        if majority_count >= self.hold and majority != self.confirmed:
            self.confirmed = majority
            self.hold_counter = 0

        return self.confirmed


# ─────────────────────────────────────────────
# TIKTOK CONTROLLER (PHASE 2)
# ─────────────────────────────────────────────

class TikTokController:
    """
    Eksekusi aksi TikTok di browser berdasarkan gesture.

    Mapping:
        SCROLL UP   → ArrowUp   (video sebelumnya)
        SCROLL DOWN → ArrowDown (video berikutnya)
        LIKE        → L         (shortcut like TikTok web)
        STOP        → tidak ada aksi
    """
    def __init__(self):
        self.last_action_time: float = 0.0
        self.last_gesture     = "UNKNOWN"
        self.cooldown         = 1.2

    def execute(self, gesture):
        now = time.time()

        if gesture == self.last_gesture:
            return
        if gesture in ("UNKNOWN", "FIST", "STOP"):
            self.last_gesture = gesture
            return
        if now - self.last_action_time < self.cooldown:
            return

        if gesture == "SCROLL UP":
            pyautogui.press("up")
            print(f"\n[TIKTOK] ↑ Video sebelumnya")

        elif gesture == "SCROLL DOWN":
            pyautogui.press("down")
            print(f"\n[TIKTOK] ↓ Video berikutnya")

        elif gesture == "LIKE":
            pyautogui.press("l")
            print(f"\n[TIKTOK] ♥ Like!")

        self.last_action_time = now
        self.last_gesture     = gesture


# ─────────────────────────────────────────────
COLORS = {
    "SCROLL UP":   (50,  220, 100),
    "SCROLL DOWN": (50,  120, 255),
    "STOP":        (50,  50,  220),
    "LIKE":        (0,   200, 255),
    "FIST":        (180, 180, 50),
    "UNKNOWN":     (150, 150, 150),
}

GESTURE_ICONS = {
    "SCROLL UP":   "^  SCROLL UP",
    "SCROLL DOWN": "v  SCROLL DOWN",
    "STOP":        "[  ]  STOP",
    "LIKE":        "LIKE  (thumbs up)",
    "FIST":        "{ }  FIST",
    "UNKNOWN":     "?  UNKNOWN",
}


def draw_ui(frame, gesture, confidence, fps, hand_count):
    h, w = frame.shape[:2]
    color = COLORS.get(gesture, (150, 150, 150))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "GESTURE TIKTOK CONTROLLER - PHASE 2",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Hands: {hand_count}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    box_h = 120
    box_y = h - box_h - 20
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (20, box_y), (w - 20, h - 20), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (20, box_y), (w - 20, h - 20), color, 2)

    label = GESTURE_ICONS.get(gesture, gesture)
    cv2.putText(frame, label,
                (50, box_y + 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2, cv2.LINE_AA)

    badge_colors = {"HIGH": (50, 200, 50), "MEDIUM": (50, 180, 220), "LOW": (100, 100, 100)}
    bc = badge_colors.get(confidence, (100, 100, 100))
    cv2.putText(frame, f"Confidence: {confidence}",
                (50, box_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q: Quit  |  S: Screenshot",
                (w - 280, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    return frame


def draw_landmarks_custom(frame, hand_landmarks, handedness):
    """Draw landmarks dengan warna custom berdasarkan tangan kiri/kanan."""
    label = handedness.classification[0].label  # 'Left' atau 'Right'
    color = (100, 220, 100) if label == "Right" else (100, 100, 220)

    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
    )


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GESTURE RECOGNITION - PHASE 2 (TikTok Controller)")
    print("  Gestures: SCROLL UP | SCROLL DOWN | LIKE | STOP")
    print("  Pastikan window TikTok aktif di browser!")
    print("  Press Q to quit, S for screenshot")
    print("=" * 60)

    if not load_model():
        print("[ERROR] Jalankan train_model.py dulu!")
        return

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka kamera. Periksa camera_index di CONFIG.")
        return

    smoother   = GestureSmoother(
        window=CONFIG["smoothing_window"],
        hold=CONFIG["gesture_hold_frames"]
    )
    controller = TikTokController()

    fps_counter    = deque(maxlen=30)
    screenshot_count: int = 0

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=CONFIG["detection_confidence"],
        min_tracking_confidence=CONFIG["tracking_confidence"],
        max_num_hands=1,
    ) as hands:

        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame tidak terbaca dari kamera.")
                break

            frame = cv2.flip(frame, 1)

            now = time.time()
            fps_counter.append(1.0 / max(now - prev_time, 1e-6))
            fps = np.mean(fps_counter)
            prev_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            gesture    = "UNKNOWN"
            confidence = "LOW"
            hand_count = 0

            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)

                for hand_lm, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    draw_landmarks_custom(frame, hand_lm, handedness)

                    raw_gesture, confidence = classify_gesture(hand_lm.landmark)
                    gesture = smoother.update(raw_gesture)
                    controller.execute(gesture)

                    print(f"\r  Gesture: {gesture:<15} | Confidence: {confidence:<6} | FPS: {fps:.1f}", end="")

            else:
                gesture = smoother.update("UNKNOWN")

            frame = draw_ui(frame, gesture, confidence, fps, hand_count)

            preview = cv2.resize(frame, (400, 225))
            cv2.imshow("Gesture TikTok Controller - Phase 2", preview)
            cv2.setWindowProperty("Gesture TikTok Controller - Phase 2",
                                  cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n\n[INFO] Program dihentikan.")
                break
            elif key == ord('s'):
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n[INFO] Screenshot disimpan: {filename}")
                screenshot_count = screenshot_count + 1  # pyre-ignore[58]

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


if __name__ == "__main__":
    main()