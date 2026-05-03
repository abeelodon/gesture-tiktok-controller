# 🤚 Gesture Recognition TikTok Controller

> Kontrol TikTok di browser menggunakan gerakan tangan — tanpa menyentuh keyboard atau mouse.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN-red)

---

## 📌 Tentang Proyek

Proyek ini membangun sistem **gesture recognition real-time** berbasis computer vision yang memungkinkan pengguna mengontrol TikTok Web menggunakan gerakan tangan di depan kamera.

Sistem menggunakan **MediaPipe** untuk mendeteksi 21 landmark tangan, lalu mengklasifikasikan gesture menggunakan model **K-Nearest Neighbors (KNN)** yang dilatih dari data tangan pengguna sendiri — sehingga akurasi optimal untuk setiap individu.

---

## ✨ Fitur

| Gesture | Aksi |
|---|---|
| ☝️ Telunjuk ke atas | Video sebelumnya |
| 👇 Telunjuk ke bawah | Video berikutnya |
| 👍 Thumbs up | Like video |
| ✋ Telapak terbuka | Stop / idle |

- **Real-time detection** — kamera berjalan di 20+ FPS
- **Always-on-top preview** — window kamera kecil tetap terlihat saat browser aktif
- **ML-based classifier** — KNN dilatih dari data tangan pengguna sendiri (bukan rule-based)
- **Anti-flicker smoother** — majority voting buffer 15 frame untuk stabilitas gesture
- **Confidence indicator** — HIGH / MEDIUM / LOW per prediksi

---

## 🏗️ Arsitektur Sistem

```
Webcam
  │
  ▼
Frame Capture (OpenCV)
  │
  ▼
Hand Detection (MediaPipe Hands)
  │  21 landmark (x, y) per tangan
  ▼
Feature Extraction
  │  42 fitur, dinormalisasi relatif ke wrist
  ▼
KNN Classifier (scikit-learn)
  │  K=3, Euclidean distance, StandardScaler
  ▼
Gesture Smoother
  │  Majority voting, window=15 frame
  ▼
TikTok Controller (pyautogui)
  │  Arrow keys + shortcut L
  ▼
Browser (TikTok Web)
```

---

## 📁 Struktur File

```
Scroll CV Project/
│
├── gesture_recognition.py   # Program utama — jalankan ini
├── collect_data.py          # Script kumpul data training
├── train_model.py           # Script training KNN model
│
├── gesture_data.csv         # Dataset landmark tangan (1000 samples)
├── gesture_model.pkl        # Model KNN terlatih (siap pakai)
│
└── README.md
```

---

## ⚙️ Instalasi

### 1. Clone / download project

```bash
git clone https://github.com/username/gesture-tiktok-controller
cd gesture-tiktok-controller
```

### 2. Buat virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe==0.10.14 numpy pyautogui scikit-learn pandas
```

---

## 🚀 Cara Pakai

### Jalankan program

```bash
python gesture_recognition.py
```

1. Buka **TikTok Web** di browser
2. Klik sekali di area video agar browser mendapat fokus
3. Lakukan gesture di depan kamera
4. Window preview kecil akan selalu terlihat di atas browser

### Controls

| Tombol | Fungsi |
|---|---|
| `Q` | Keluar |
| `S` | Screenshot |

---

## 🤖 Training Model Sendiri

Model dilatih dari data tangan pengguna sendiri untuk akurasi optimal.

### 1. Kumpulkan data

```bash
python collect_data.py
```

| Tombol | Gesture | Target |
|---|---|---|
| `1` | ☝️ SCROLL UP | 200 sample |
| `2` | 👇 SCROLL DOWN | 200 sample |
| `3` | 👍 LIKE | 200 sample |
| `4` | ✋ STOP | 200 sample |
| `5` | 🤚 UNKNOWN (random) | 200 sample |

### 2. Train model

```bash
python train_model.py
```

Output: `gesture_model.pkl` siap dipakai.

---

## 📊 Performa Model

| Metric | Nilai |
|---|---|
| Total training data | 1.000 samples |
| Algoritma | K-Nearest Neighbors (K=3) |
| Test accuracy | 98.5% |
| Normalisasi | StandardScaler |
| Metrik jarak | Euclidean |

Model divalidasi dengan train/test split 80:20 dan 5-fold cross validation.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **OpenCV** — video capture & rendering
- **MediaPipe** — hand tracking & landmark extraction
- **scikit-learn** — KNN classifier & preprocessing
- **pyautogui** — keyboard automation
- **NumPy / Pandas** — data processing

---

## 🔮 Pengembangan Selanjutnya

- [ ] Integrasi Android via ADB
- [ ] Gesture tambahan (follow, share)
- [ ] Support multi-tangan
- [ ] Web dashboard monitoring gesture

---

## 👤 Author

**[Nama Kamu]**
- GitHub: [@username](https://github.com/username)
- LinkedIn: [linkedin.com/in/username](https://linkedin.com/in/username)