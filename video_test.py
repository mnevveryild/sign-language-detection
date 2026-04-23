"""
═══════════════════════════════════════════════════════════════════════════════
İşaret Dili Tanıma — Tek Video Testi
═══════════════════════════════════════════════════════════════════════════════

Bir video dosyası alır, eğitim pipeline'ıyla birebir aynı şekilde işler,
Top-5 tahmin sonucunu ekranda gösterir.

KULLANIM:
  python video_test.py "C:\\yol\\video.mp4"

  Veya argüman vermeden çalıştırırsan dosya yolu ister.

GEREKLİLİKLER:
  pip install opencv-python mediapipe numpy tensorflow

DOSYA YAPISI:
  calisma_klasoru/
  ├── video_test.py                   ← bu dosya
  ├── best_model_filtered.keras
  ├── demo_assets_filtered/
  │   ├── norm_stats.json
  │   └── label_encoder_classes.npy
  ├── hand_landmarker.task            ← ilk çalıştırmada iner
  └── pose_landmarker.task
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
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

SEQ_LEN    = 30
FEAT_DIM   = 165
MIN_FRAMES = 8
POSE_UPPER_BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

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


# ═══════════════════════════════════════════════════════════════════════════════
# VİDEO → HAM LANDMARK LİSTESİ (extract_all_npy.ipynb'deki mantık)
# ═══════════════════════════════════════════════════════════════════════════════
def process_video(video_path):
    """Tek video'dan ham landmark frame'leri çıkar (eğitim koduyla aynı)."""
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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose_det.close()
        hand_det.close()
        return None

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames    = []
    frame_idx = 0

    print(f"  Video FPS     : {fps:.1f}")
    print(f"  Toplam frame  : {total}")
    print(f"  Süre          : {total/fps:.2f} sn")
    print(f"  Landmark çıkarılıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(frame_idx * 1000 / fps)

        # Pose
        pose_res = pose_det.detect_for_video(mp_img, ts_ms)
        pose_data = None
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            pose_data = []
            for idx in POSE_UPPER_BODY:
                pose_data.extend([lms[idx].x, lms[idx].y, lms[idx].z])

        # Hands
        hand_res = hand_det.detect_for_video(mp_img, ts_ms)
        left_hand, right_hand = None, None
        if hand_res.hand_landmarks and hand_res.handedness:
            for lm_list, hd_list in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = hd_list[0].category_name
                pts   = []
                for p in lm_list:
                    pts.extend([p.x, p.y, p.z])
                if label == "Left":
                    left_hand = pts
                else:
                    right_hand = pts

        frames.append({"pose": pose_data,
                       "left_hand": left_hand,
                       "right_hand": right_hand})
        frame_idx += 1

    cap.release()
    pose_det.close()
    hand_det.close()
    return frames


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
# ÖN İŞLEME
# ═══════════════════════════════════════════════════════════════════════════════
def center_landmarks(seq):
    centered = seq.copy()
    for t in range(seq.shape[0]):
        nose_x = seq[t, 0]
        nose_y = seq[t, 1]
        centered[t, 0::3] = seq[t, 0::3] - nose_x
        centered[t, 1::3] = seq[t, 1::3] - nose_y
    return centered


def extract_active_frames(seq, n_frames=30):
    hand_coords = seq[:, 39:]
    deltas      = np.sum(np.abs(np.diff(hand_coords, axis=0)), axis=1)

    if len(deltas) == 0 or deltas.max() < 1e-6:
        return seq

    peak  = int(np.argmax(deltas))
    half  = n_frames // 2
    start = max(0, peak - half)
    end   = start + n_frames

    if end > len(seq):
        end   = len(seq)
        start = max(0, end - n_frames)

    active_seq = seq[start:end]
    if len(active_seq) < n_frames:
        pad        = np.zeros((n_frames - len(active_seq), seq.shape[1]), dtype=seq.dtype)
        active_seq = np.vstack([active_seq, pad])

    return active_seq


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL + ASSET YÜKLE
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
    return model, f_mean, f_std, classes


# ═══════════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════════
def predict_video(video_path, model, f_mean, f_std, classes):
    """Tek video için tahmin yap ve Top-5 sonucu döndür."""
    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {video_path}")
        return None

    print("\n" + "="*60)
    print(f"  VIDEO: {os.path.basename(video_path)}")
    print("="*60)

    # 1) Landmark çıkar
    start = time.time()
    raw_frames = process_video(video_path)
    extract_time = time.time() - start

    if raw_frames is None:
        print("❌ Video okunamadı")
        return None

    if len(raw_frames) < MIN_FRAMES:
        print(f"❌ Çok az frame: {len(raw_frames)} (minimum {MIN_FRAMES})")
        return None

    print(f"  ✓ Landmark çıkarıldı ({extract_time:.1f} sn)")

    # 2) Pipeline
    seq = resample_frames(raw_frames, SEQ_LEN)       # (30, 165)
    #seq = extract_active_frames(seq, SEQ_LEN)
    #seq = center_landmarks(seq)
    seq = (seq - f_mean) / f_std
    X   = seq.astype(np.float32)[np.newaxis, ...]    # (1, 30, 165)

    # 3) Tahmin
    start = time.time()
    probs = model.predict(X, verbose=0)[0]
    pred_time = time.time() - start

    # 4) Top-5
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [(classes[i], float(probs[i])) for i in top5_idx]

    print(f"  ✓ Tahmin tamam ({pred_time*1000:.0f} ms)")
    print("\n" + "─"*60)
    print("  TAHMİNLER:")
    print("─"*60)
    for i, (name, p) in enumerate(top5, 1):
        bar_len = int(p * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {i}. {name:20s} {bar} %{p*100:5.1f}")
    print("─"*60)

    return top5


def main():
    # Video yolunu al
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Test etmek istediğin videonun tam yolunu gir:")
        print("  (Örnek: C:\\Users\\...\\kitap_001_color.mp4)")
        video_path = input("\nVideo yolu: ").strip().strip('"').strip("'")

    if not video_path:
        print("❌ Video yolu verilmedi")
        return

    # Kurulumlar
    ensure_models()
    model, f_mean, f_std, classes = load_model_and_assets()

    # Tahmin
    predict_video(video_path, model, f_mean, f_std, classes)

    # Tekrar test seçeneği
    print("\nBaşka video test etmek ister misin? (E/H)")
    while input("> ").strip().lower() in ('e', 'evet'):
        print("\nYeni video yolu:")
        path = input("> ").strip().strip('"').strip("'")
        if path:
            predict_video(path, model, f_mean, f_std, classes)
        print("\nDevam etmek ister misin? (E/H)")


if __name__ == "__main__":
    main()
