"""
═══════════════════════════════════════════════════════════════════════════════
İşaret Dili Tanıma — Kamera Testi (SPACE ile Manuel Tetikleme)
═══════════════════════════════════════════════════════════════════════════════

20 sınıflık demo model ile kameradan canlı test.

KULLANIM:
  SPACE : Kayıt başlat (3 saniye otomatik biter), tekrar basarak erken bitir
  R     : Sonucu temizle
  Q     : Çıkış

GEREKLİLİKLER:
  pip install opencv-python mediapipe numpy tensorflow

DOSYA YAPISI:
  calisma_klasoru/
  ├── kamera_testi.py                 ← bu dosya
  ├── best_model_demo20.keras
  ├── demo_assets/
  │   ├── norm_stats.json
  │   └── label_encoder_classes.npy
  ├── hand_landmarker.task            ← ilk çalıştırmada iner
  └── pose_landmarker.task

20 SINIF:
  kitap, okul, sapka, ayakkabi, kopek, kardes, arkadas, tavan, yildiz, calismak,
  patates, kolonya, kopru, kapi, pencere, semsiye, havlu, cekic, tesekkur, iyi
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import json
import time
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
from tensorflow import keras

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH         = "best_model_demo20.keras"
NORM_STATS_PATH    = "demo_assets/norm_stats.json"
LABEL_CLASSES_PATH = "demo_assets/label_encoder_classes.npy"
HAND_MODEL_PATH    = "hand_landmarker.task"
POSE_MODEL_PATH    = "pose_landmarker.task"

# --- Eğitimdeki sabitler ---
SEQ_LEN    = 30
FEAT_DIM   = 165
MIN_FRAMES = 8
POSE_UPPER_BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# --- Kayıt süresi ---
RECORD_DURATION_SEC = 3.0
TARGET_FPS          = 30

# --- Model URL'leri ---
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM LOSS
# ═══════════════════════════════════════════════════════════════════════════════
class SparseLabelSmoothingLoss(keras.losses.Loss):
    def __init__(self, num_classes, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.squeeze(y_true), tf.int32)
        one_hot = tf.one_hot(y_true, self.num_classes)
        smooth  = one_hot * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0)
        loss    = -tf.reduce_sum(smooth * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_classes': self.num_classes, 'smoothing': self.smoothing})
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DOSYALARINI İNDİR
# ═══════════════════════════════════════════════════════════════════════════════
def ensure_models():
    for url, path in [(HAND_MODEL_URL, HAND_MODEL_PATH),
                      (POSE_MODEL_URL, POSE_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"İndiriliyor: {path} ...")
            urllib.request.urlretrieve(url, path)
    print("✓ MediaPipe modelleri hazır")


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK ÇIKARMA
# ═══════════════════════════════════════════════════════════════════════════════
def create_detectors():
    """VIDEO running mode'da iki detector oluştur (eğitim koduyla aynı)."""
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_det = mp_vision.PoseLandmarker.create_from_options(pose_options)
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_options)
    return pose_det, hand_det


def extract_frame_landmarks(frame_bgr, pose_det, hand_det, ts_ms):
    """Tek frame'den eğitim formatında dict döndür."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Pose
    pose_data = None
    pose_res  = pose_det.detect_for_video(mp_img, ts_ms)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        pose_data = []
        for idx in POSE_UPPER_BODY:
            pose_data.extend([lms[idx].x, lms[idx].y, lms[idx].z])

    # Hands
    left_hand, right_hand = None, None
    hand_res = hand_det.detect_for_video(mp_img, ts_ms)
    if hand_res.hand_landmarks and hand_res.handedness:
        for lm_list, hd_list in zip(hand_res.hand_landmarks, hand_res.handedness):
            label = hd_list[0].category_name
            pts = []
            for p in lm_list:
                pts.extend([p.x, p.y, p.z])
            if label == "Left":
                left_hand = pts
            else:
                right_hand = pts

    return {"pose": pose_data, "left_hand": left_hand, "right_hand": right_hand}


# ═══════════════════════════════════════════════════════════════════════════════
# RESAMPLE (eğitim koduyla birebir)
# ═══════════════════════════════════════════════════════════════════════════════
def resample_frames(frames, target=SEQ_LEN):
    n = len(frames)
    if n == 0:
        return np.zeros((target, FEAT_DIM), dtype=np.float32)

    indices   = np.linspace(0, n - 1, target)
    resampled = []

    for idx in indices:
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        t = idx - lo
        fv = []

        pa, pb = frames[lo].get("pose"), frames[hi].get("pose")
        if pa is None and pb is None:
            fv.extend([0.0] * 39)
        elif pa is None:
            fv.extend(pb)
        elif pb is None:
            fv.extend(pa)
        else:
            fv.extend((np.array(pa) * (1 - t) + np.array(pb) * t).tolist())

        for hand in ("left_hand", "right_hand"):
            a, b = frames[lo].get(hand), frames[hi].get(hand)
            if a is None and b is None:
                fv.extend([0.0] * 63)
            elif a is None:
                fv.extend(b)
            elif b is None:
                fv.extend(a)
            else:
                fv.extend((np.array(a) * (1 - t) + np.array(b) * t).tolist())

        resampled.append(fv)

    return np.array(resampled, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE — video_test.py'deki çalışan versiyonla aynı (sadece resample + norm)
# ═══════════════════════════════════════════════════════════════════════════════
def process_recording(raw_frames, f_mean, f_std):
    """Ham frame dict listesini → (1, 30, 165) normalize tensöre çevir."""
    seq = resample_frames(raw_frames, SEQ_LEN)   # (30, 165)
    seq = (seq - f_mean) / f_std
    return seq.astype(np.float32)[np.newaxis, ...]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL + NORM STATS YÜKLE
# ═══════════════════════════════════════════════════════════════════════════════
def load_model_and_assets():
    print("Model yükleniyor...")
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'SparseLabelSmoothingLoss': SparseLabelSmoothingLoss}
    )

    with open(NORM_STATS_PATH, 'r') as f:
        ns = json.load(f)
    f_mean = np.array(ns['mean'], dtype=np.float32)
    f_std  = np.array(ns['std'],  dtype=np.float32)

    classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)

    print(f"✓ Model  : {model.count_params():,} parametre")
    print(f"✓ Sınıf  : {len(classes)}")
    print(f"✓ Kelimeler: {', '.join(classes)}")
    return model, f_mean, f_std, classes


# ═══════════════════════════════════════════════════════════════════════════════
# TAHMİN YAP
# ═══════════════════════════════════════════════════════════════════════════════
def predict(recorded_frames, model, f_mean, f_std, classes):
    X = process_recording(recorded_frames, f_mean, f_std)
    probs = model.predict(X, verbose=0)[0]

    top3_idx   = np.argsort(probs)[-3:][::-1]
    top3_names = [classes[i] for i in top3_idx]
    top3_probs = [float(probs[i]) for i in top3_idx]

    return {'top3': list(zip(top3_names, top3_probs)), 'time': time.time()}


def print_prediction(prediction):
    print("\n📊 TAHMİNLER:")
    for i, (name, p) in enumerate(prediction['top3'], 1):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"  {i}. {name:15s} {bar} %{p*100:5.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANA DÖNGÜ
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ensure_models()
    model, f_mean, f_std, classes = load_model_and_assets()
    pose_det, hand_det = create_detectors()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    # Durum
    is_recording      = False
    recording_start   = 0.0
    recorded_frames   = []
    record_frame_idx  = 0
    last_prediction   = None

    print("\n" + "="*60)
    print("  KAMERA HAZIR")
    print("="*60)
    print("  SPACE : Kayıt başlat/durdur (3 saniye otomatik biter)")
    print("  R     : Sonucu temizle")
    print("  Q     : Çıkış")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Kayıt sırasında landmark topla ──
        if is_recording:
            elapsed = time.time() - recording_start

            ts_ms = int(record_frame_idx * 1000 / TARGET_FPS)
            landmarks = extract_frame_landmarks(frame, pose_det, hand_det, ts_ms)
            recorded_frames.append(landmarks)
            record_frame_idx += 1

            if elapsed >= RECORD_DURATION_SEC:
                is_recording = False
                print(f"\n⏹  Kayıt bitti: {len(recorded_frames)} frame")

                if len(recorded_frames) < MIN_FRAMES:
                    print(f"⚠️  Çok az frame")
                else:
                    last_prediction = predict(recorded_frames, model, f_mean, f_std, classes)
                    print_prediction(last_prediction)

                recorded_frames = []

            # Kayıt göstergesi
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            remaining = RECORD_DURATION_SEC - elapsed
            cv2.putText(frame, f"REC {remaining:.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, f"frames: {len(recorded_frames)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "SPACE: kayit baslat", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Q: cikis", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── Son tahmini ekrana yaz ──
        if last_prediction is not None:
            overlay_h = 140
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, "TAHMIN:", (20, h - overlay_h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for i, (name, p) in enumerate(last_prediction['top3']):
                y = h - overlay_h + 55 + i * 28
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                text  = f"{i+1}. {name}  %{p*100:.1f}"
                cv2.putText(frame, text, (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Isaret Dili Tanima - 20 Sinif", frame)

        # ── Tuş kontrolü ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif key == ord(' '):
            if not is_recording:
                is_recording     = True
                recording_start  = time.time()
                recorded_frames  = []
                record_frame_idx = 0
                last_prediction  = None
                print(f"\n🔴 Kayıt başladı ({RECORD_DURATION_SEC} saniye)...")
            else:
                is_recording = False
                print(f"\n⏹  Kayıt manuel bitirildi: {len(recorded_frames)} frame")

                if len(recorded_frames) < MIN_FRAMES:
                    print(f"⚠️  Çok az frame")
                else:
                    last_prediction = predict(recorded_frames, model, f_mean, f_std, classes)
                    print_prediction(last_prediction)

                recorded_frames = []

        elif key == ord('r') or key == ord('R'):
            last_prediction = None
            print("✓ Tahmin temizlendi")

    cap.release()
    cv2.destroyAllWindows()
    pose_det.close()
    hand_det.close()
    print("\nÇıkış yapıldı.")


if __name__ == "__main__":
    main()
