"""
╔══════════════════════════════════════════════════════════════╗
║   DATA COLLECTION - Gesture ML Training                     ║
╚══════════════════════════════════════════════════════════════╝

Cara pakai:
    python collect_data.py

Controls:
    1 = Rekam SCROLL_UP   (☝️ telunjuk ke atas)
    2 = Rekam SCROLL_DOWN (👇 telunjuk ke bawah)
    3 = Rekam LIKE        (👍 thumbs up)
    4 = Rekam STOP        (✋ telapak terbuka)
    5 = Rekam UNKNOWN     (tangan santai / posisi random)
    SPACE = pause / resume
    Q = simpan & keluar
"""

import cv2
import mediapipe as mp
import csv
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_CSV       = "gesture_data.csv"
TARGET_PER_LABEL = 1000
CAPTURE_DELAY    = 3  # Ambil 1 sample tiap N frame — lebih lambat, lebih variatif

LABELS = {
    ord('1'): "SCROLL_UP",
    ord('2'): "SCROLL_DOWN",
    ord('3'): "LIKE",
    ord('4'): "STOP",
    ord('5'): "UNKNOWN",
}

COLORS = {
    "SCROLL_UP":   (50,  220, 100),
    "SCROLL_DOWN": (50,  120, 255),
    "LIKE":        (0,   200, 255),
    "STOP":        (50,  50,  220),
    "UNKNOWN":     (150, 150, 150),
}

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(landmarks):
    """
    42 fitur: (x, y) tiap 21 landmark dinormalisasi
    relatif ke wrist dan skala tangan.
    """
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

# ─────────────────────────────────────────────
# LOAD EXISTING COUNT
# ─────────────────────────────────────────────

def load_counts():
    counts = {v: 0 for v in LABELS.values()}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[-1] in counts:
                    counts[row[-1]] += 1
    return counts

# ─────────────────────────────────────────────
# DRAW UI
# ─────────────────────────────────────────────

def draw_ui(frame, counts, active_label, recording, hand_detected):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    status = "● REC" if recording else "■ PAUSE"
    color  = (50, 50, 220) if recording else (150, 150, 150)
    cv2.putText(frame, f"DATA COLLECTION  |  {status}",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    bar_y = 60
    for label, count in counts.items():
        pct   = min(count / TARGET_PER_LABEL, 1.0)
        bar_w = int((w - 30) * pct)
        col   = COLORS.get(label, (150, 150, 150))
        is_active = (label == active_label)

        bg_col = (30, 30, 30) if is_active else (15, 15, 15)
        cv2.rectangle(frame, (15, bar_y), (w - 15, bar_y + 28), bg_col, -1)
        if bar_w > 0:
            cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + 28), col, -1)
        cv2.rectangle(frame, (15, bar_y), (w - 15, bar_y + 28),
                      col if is_active else (50, 50, 50), 1)

        key_num = [k for k, v in LABELS.items() if v == label]
        key_str = chr(key_num[0]) if key_num else "?"
        cv2.putText(frame, f"[{key_str}] {label:<14} {count:>3}/{TARGET_PER_LABEL}",
                    (20, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255) if is_active else (160, 160, 160), 1, cv2.LINE_AA)
        bar_y += 36

    if active_label:
        col      = COLORS.get(active_label, (150, 150, 150))
        rec_text = f"{'RECORDING' if recording else 'PAUSED'}  ->  {active_label}"
        cv2.rectangle(frame, (0, h - 60), (w, h), (10, 10, 10), -1)
        cv2.rectangle(frame, (0, h - 60), (w, h), col, 2)
        cv2.putText(frame, rec_text,
                    (20, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2, cv2.LINE_AA)

    dot_col = (50, 220, 50) if hand_detected else (80, 80, 80)
    cv2.circle(frame, (w - 25, 25), 8, dot_col, -1)

    cv2.putText(frame, "1-5: pilih label  |  SPACE: pause  |  Q: simpan & keluar",
                (15, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return frame

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  DATA COLLECTION - Gesture ML")
    print("  Tekan 1-5 untuk pilih gesture, lakukan gesture,")
    print("  tekan SPACE untuk pause, Q untuk selesai")
    print("=" * 55)

    counts       = load_counts()
    active_label = None
    recording    = False
    frame_count  = 0  # counter untuk delay antar sample

    file_exists = os.path.exists(OUTPUT_CSV)
    csvfile = open(OUTPUT_CSV, "a", newline="")
    writer  = csv.writer(csvfile)
    if not file_exists:
        header = [f"f{i}" for i in range(42)] + ["label"]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.flip(frame, 1)
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results      = hands.process(rgb)
            rgb.flags.writeable = True
            hand_detected = False
            frame_count  += 1

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_lm = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(100, 220, 100), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

                # Rekam 1 sample tiap CAPTURE_DELAY frame
                if recording and active_label and (frame_count % CAPTURE_DELAY == 0):
                    features = extract_features(hand_lm.landmark)
                    writer.writerow(features + [active_label])
                    csvfile.flush()
                    counts[active_label] += 1

                    if counts[active_label] >= TARGET_PER_LABEL:
                        recording = False
                        print(f"\n[INFO] {active_label} selesai! ({TARGET_PER_LABEL} samples)")

            frame = draw_ui(frame, counts, active_label, recording, hand_detected)

            preview = cv2.resize(frame, (800, 450))
            cv2.imshow("Data Collection", preview)
            cv2.setWindowProperty("Data Collection", cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if active_label:
                    recording = not recording
                    print(f"\n[INFO] {'REC' if recording else 'PAUSE'} - {active_label}")
            elif key in LABELS:
                active_label = LABELS[key]
                recording    = True
                print(f"\n[INFO] Mulai rekam: {active_label} ({counts[active_label]}/{TARGET_PER_LABEL})")

    csvfile.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print("  SELESAI - Ringkasan data:")
    for label, count in counts.items():
        done = "DONE" if count >= TARGET_PER_LABEL else f"{count}/{TARGET_PER_LABEL}"
        print(f"  {label:<14} {done}")
    print(f"\n  Disimpan ke: {OUTPUT_CSV}")
    print("  Jalankan: python train_model.py")
    print("=" * 55)


if __name__ == "__main__":
    main()