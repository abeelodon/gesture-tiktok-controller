"""
╔══════════════════════════════════════════════════════════════╗
║   TRAIN MODEL - KNN Gesture Classifier                      ║
╚══════════════════════════════════════════════════════════════╝

Jalankan setelah collect_data.py selesai:
    python train_model.py

Output:
    gesture_model.pkl  — model siap pakai di gesture_recognition.py
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

CSV_FILE   = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

def main():
    print("=" * 55)
    print("  TRAINING KNN GESTURE MODEL")
    print("=" * 55)

    # ── Load data ──
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] {CSV_FILE} tidak ditemukan!")
        print("  Jalankan collect_data.py dulu.")
        return

    df = pd.read_csv(CSV_FILE)
    print(f"\n[INFO] Total data: {len(df)} samples")
    print(f"[INFO] Distribusi label:")
    print(df["label"].value_counts().to_string())

    if len(df) < 50:
        print("\n[ERROR] Data terlalu sedikit, minimal 50 samples.")
        return

    # ── Split fitur & label ──
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # ── Normalisasi ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split train/test 80:20 ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Cari K terbaik (3, 5, 7, 9) ──
    print("\n[INFO] Mencari K terbaik...")
    best_k, best_score = 5, 0
    for k in [3, 5, 7, 9]:
        knn    = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        mean   = scores.mean()
        print(f"  K={k}  CV accuracy: {mean:.3f}")
        if mean > best_score:
            best_score = mean
            best_k     = k

    print(f"\n[INFO] K terbaik: {best_k} (CV accuracy: {best_score:.3f})")

    # ── Train model final ──
    model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    model.fit(X_train, y_train)

    # ── Evaluasi ──
    y_pred    = model.predict(X_test)
    test_acc  = (y_pred == y_test).mean()
    print(f"\n[INFO] Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")

    print("\n[INFO] Classification report:")
    print(classification_report(y_test, y_pred))

    print("[INFO] Confusion matrix:")
    labels = sorted(set(y))
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    header = f"{'':>14}" + "".join(f"{l:>14}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>14}" + "".join(f"{v:>14}" for v in row))

    # ── Simpan model + scaler ──
    bundle = {"model": model, "scaler": scaler, "labels": labels}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[INFO] Model disimpan ke: {MODEL_FILE}")

    if test_acc >= 0.85:
        print("  Status: SIAP dipakai di gesture_recognition.py")
    else:
        print("  Status: Akurasi kurang dari 85%")
        print("  Tips: tambah lebih banyak sample, terutama gesture yang sering salah")

    print("=" * 55)


if __name__ == "__main__":
    main()