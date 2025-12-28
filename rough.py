#!/usr/bin/env python3
# ==============================================================================
# DEEPFAKE DETECTION: rPPG + FOURIER TRANSFORM (Final Complete)
# Based on Yin et al. (2024)
# ==============================================================================
# STATUS: RESEARCH GRADE
# [x] Robust Signal Processing (CHROM + Zero-Pad FFT)
# [x] Leakage-Proof Data Split
# [x] Fixed Keras Layer Lifecycle (No Crashes)
# [x] Corrupt Video Handling (New)
# [x] Full File Logging (New)
# ==============================================================================

import os
import cv2
import sys
import json
import random
import numpy as np
import mediapipe as mp
import scipy.signal as signal
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, optimizers, callbacks

# ==============================================================================
# 1. GLOBAL CONFIG & SEEDING
# ==============================================================================
CONFIG = {
    # Data Paths - ADJUST THESE TO YOUR CLUSTER PATHS
    'REAL_PATH': "dataset_face_forensics_plus_plus/real_sequences",
    'FAKE_PATH': "dataset_face_forensics_plus_plus/fake_sequences",
    'CACHE_DIR': "processed_data_cache_final",
    'RESULTS_DIR': "final_model_results",
    
    # Preprocessing
    'FPS': 25,
    'WINDOW_SEC': 3,
    'STRIDE_SEC': 3,         # Non-overlapping for max diversity
    'NUM_ROIS': 22,
    'IMG_SIZE': 240,
    
    # Frequency Masking (Physiological Range)
    'MIN_FREQ': 0.75,        # 45 BPM
    'MAX_FREQ': 4.0,         # 240 BPM
    
    # Training
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'LR': 1e-4,
    'SEED': 42
}

def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(CONFIG['SEED'])

# GPU Config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        print(f"GPU Detected: {len(gpus)}")
    except RuntimeError as e: print(e)

os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)

# ==============================================================================
# 2. ROBUST SIGNAL PROCESSING
# ==============================================================================

def sliding_window_detrend(signal_data, window_size=15):
    """Removes low-frequency drift."""
    detrended = np.zeros_like(signal_data)
    for i in range(len(signal_data)):
        start = max(0, i - window_size // 2)
        end = min(len(signal_data), i + window_size // 2 + 1)
        detrended[i] = signal_data[i] - np.mean(signal_data[start:end])
    return detrended

def butterworth_bandpass(signal_data, lowcut=0.8, highcut=3.0, fs=25, order=2):
    """Standard physiological bandpass."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)

def get_chrom_signal(frames_rgb):
    """
    CHROM Method (Robust Version).
    Applies filtering BEFORE alpha calculation to remove noise/drift influence.
    """
    # 1. Raw Projection
    X = 3 * frames_rgb[:, 0] - 2 * frames_rgb[:, 1]
    Y = 1.5 * frames_rgb[:, 0] + frames_rgb[:, 1] - 1.5 * frames_rgb[:, 2]

    # 2. Filter Projections
    Xf = butterworth_bandpass(X, fs=CONFIG['FPS'])
    Yf = butterworth_bandpass(Y, fs=CONFIG['FPS'])

    # 3. Calculate Alpha (Robust)
    std_x = np.std(Xf)
    std_y = np.std(Yf)
    alpha = std_x / (std_y + 1e-6)

    # 4. Combine
    S = Xf - alpha * Yf
    
    # 5. Final Detrend & Normalize
    S = sliding_window_detrend(S)
    S_norm = (S - np.mean(S)) / (np.std(S) + 1e-6)
    return S_norm

def get_masked_fft(signal_data):
    """
    FFT with Physiological Masking + Zero Padding.
    """
    N = len(signal_data)
    fft_complex = np.fft.rfft(signal_data)
    fft_mag = np.abs(fft_complex)
    freqs = np.fft.rfftfreq(N, d=1/CONFIG['FPS'])
    
    # Masking
    mask = (freqs >= CONFIG['MIN_FREQ']) & (freqs <= CONFIG['MAX_FREQ'])
    fft_masked = fft_mag * mask.astype(float)
    
    # Zero Padding to preserve bin positions relative to N
    fft_padded = np.zeros(N)
    fft_padded[:len(fft_masked)] = fft_masked
    
    # Normalize
    return (fft_padded - fft_padded.min()) / (fft_padded.max() - fft_padded.min() + 1e-6)

# ==============================================================================
# 3. FACE & ROI ENGINE
# ==============================================================================

class FaceEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
        )
        self.roi_map = {
            'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'cheek_left': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10],
            'cheek_right': [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
            'philtrum': [164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
        }

    def process_frame(self, image):
        h, w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        rois_rgb = []

        def get_sub_rois(indices, rows, cols):
            pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices], dtype=np.int32)
            x, y, w_box, h_box = cv2.boundingRect(pts)
            crop = image[y:y+h_box, x:x+w_box]
            if crop.size == 0: return [np.mean(image, axis=(0,1))] * (rows*cols)
            
            sub_h, sub_w, _ = crop.shape
            dy, dx = sub_h // rows, sub_w // cols
            means = []
            for r in range(rows):
                for c in range(cols):
                    cell = crop[r*dy:(r+1)*dy, c*dx:(c+1)*dx]
                    if cell.size == 0: 
                        means.append(np.mean(crop, axis=(0,1)))
                    else:
                        # Erosion: Center 80%
                        ch, cw, _ = cell.shape
                        cy, cx = ch//2, cw//2
                        nh, nw = int(ch*0.8), int(cw*0.8)
                        center = cell[cy-nh//2:cy+nh//2, cx-nw//2:cx+nw//2]
                        if center.size == 0: center = cell
                        means.append(np.mean(center, axis=(0,1)))
            return means

        rois_rgb.extend(get_sub_rois(self.roi_map['forehead'], 2, 3))
        rois_rgb.extend(get_sub_rois(self.roi_map['cheek_left'], 2, 3))
        rois_rgb.extend(get_sub_rois(self.roi_map['cheek_right'], 2, 3))
        rois_rgb.extend(get_sub_rois(self.roi_map['philtrum'], 2, 2))
        return np.array(rois_rgb)

# ==============================================================================
# 4. PREPROCESSING PIPELINE (With Leakage Fix & File Checks)
# ==============================================================================

def generate_dataset_split(file_list, save_name):
    """Generates MVHMs for a specific list of files (Train or Test)."""
    save_x = os.path.join(CONFIG['CACHE_DIR'], f'X_{save_name}.npy')
    save_y = os.path.join(CONFIG['CACHE_DIR'], f'y_{save_name}.npy')
    
    if os.path.exists(save_x) and os.path.exists(save_y):
        print(f"{save_name} data already exists. Skipping.")
        return

    print(f"Generating {save_name} data ({len(file_list)} videos)...")
    processor = FaceEngine()
    
    X_data, y_data = [], []
    frames_win = CONFIG['FPS'] * CONFIG['WINDOW_SEC']
    stride = CONFIG['FPS'] * CONFIG['STRIDE_SEC']
    
    skipped_corrupt = 0
    
    for vid_path, label in tqdm(file_list):
        # FIX: Check for corrupted video
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            skipped_corrupt += 1
            continue
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if len(frames) < frames_win: continue
        
        for start in range(0, len(frames) - frames_win + 1, stride):
            window = frames[start : start + frames_win]
            
            raw_sigs = []
            valid = True
            for frame in window:
                rois = processor.process_frame(frame)
                if rois is None: 
                    valid = False
                    break
                raw_sigs.append(rois)
            
            if not valid: continue
            
            raw_sigs = np.array(raw_sigs) 
            rppg_list, fft_list = [], []
            
            for i in range(CONFIG['NUM_ROIS']):
                trace = raw_sigs[:, i, :]
                rppg = get_chrom_signal(trace)      
                fft = get_masked_fft(rppg)          
                
                # Normalize 0-1 
                rppg_n = (rppg - rppg.min()) / (rppg.max() - rppg.min() + 1e-6)
                rppg_list.append(rppg_n)
                fft_list.append(fft)
                
            mat = np.vstack([np.array(rppg_list), np.array(fft_list)]) 
            mvhm = cv2.resize(mat, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']), interpolation=cv2.INTER_CUBIC)
            mvhm = np.stack([mvhm]*3, axis=-1)
            
            X_data.append(mvhm.astype(np.float16))
            y_data.append(label)
    
    print(f"Skipped {skipped_corrupt} corrupted/unopenable videos.")
    np.save(save_x, np.array(X_data))
    np.save(save_y, np.array(y_data))
    print(f"Saved {len(X_data)} samples for {save_name}.")

def run_preprocessing():
    # 1. Collect all video paths
    real_files = [os.path.join(CONFIG['REAL_PATH'], f) for f in os.listdir(CONFIG['REAL_PATH']) if f.endswith('.mp4')]
    fake_files = [os.path.join(CONFIG['FAKE_PATH'], f) for f in os.listdir(CONFIG['FAKE_PATH']) if f.endswith('.mp4')]
    
    # 2. Assign Labels (0=Real, 1=Fake)
    all_real = [(p, 0) for p in real_files]
    all_fake = [(p, 1) for p in fake_files]
    
    # 3. VIDEO-LEVEL SPLIT (Leakage Proof)
    train_real, test_real = train_test_split(all_real, test_size=0.2, random_state=CONFIG['SEED'])
    train_fake, test_fake = train_test_split(all_fake, test_size=0.2, random_state=CONFIG['SEED'])
    
    train_list = train_real + train_fake
    test_list = test_real + test_fake
    
    random.shuffle(train_list)
    random.shuffle(test_list)
    
    # 4. Generate Data
    generate_dataset_split(train_list, 'train')
    generate_dataset_split(test_list, 'test')

# ==============================================================================
# 5. TRAINING MODEL (FIXED LIFECYCLE)
# ==============================================================================

class SpatialAttention(layers.Layer):
    """
    Fixed Spatial Attention Layer:
    - Instantiates BatchNorm in __init__ (prevents tf.function crash)
    - Reuses layers in call
    - Explicitly freezes BN (trainable=False, training=False)
    """
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        
        # 1. Input Norm
        self.bn_in = layers.BatchNormalization(trainable=False)
        
        # 2. Backbone Branch
        self.conv1 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization(trainable=False)
        self.conv2 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization(trainable=False)

        # 3. Soft Mask Branch
        self.m_conv1 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.m_bn1 = layers.BatchNormalization(trainable=False)
        self.m_conv2 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.m_bn2 = layers.BatchNormalization(trainable=False)
        self.m_out = layers.Conv2D(1, 1, activation='sigmoid')

    def call(self, inputs, training=False):
        # Input Norm
        x = self.bn_in(inputs, training=False)

        # Backbone
        t = self.conv1(x)
        t = self.bn1(t, training=False)
        t = self.conv2(t)
        t = self.bn2(t, training=False)

        # Mask
        m = self.m_conv1(x)
        m = self.m_bn1(m, training=False)
        m = self.m_conv2(m)
        m = self.m_bn2(m, training=False)
        mask = self.m_out(m)

        # Modulation
        modulated = layers.Multiply()([t, mask])
        return layers.Add()([x, modulated])

def build_model():
    base = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(240, 240, 3))
    
    # Freeze Base Model + its BatchNorm
    base.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    x = base.output
    x = SpatialAttention()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(layers.Dense(1024, activation='relu')(x))
    x = layers.Dropout(0.5)(layers.Dense(512, activation='relu')(x))
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(base.input, output)
    model.compile(optimizer=optimizers.Adam(CONFIG['LR']), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_training():
    print("Loading Data...")
    X_train = np.load(os.path.join(CONFIG['CACHE_DIR'], 'X_train.npy')).astype(np.float32) / 255.0
    y_train = np.load(os.path.join(CONFIG['CACHE_DIR'], 'y_train.npy'))
    X_test = np.load(os.path.join(CONFIG['CACHE_DIR'], 'X_test.npy')).astype(np.float32) / 255.0
    y_test = np.load(os.path.join(CONFIG['CACHE_DIR'], 'y_test.npy'))
    
    # Class Weights (Window Level)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {i: cw[i] for i in range(len(cw))}
    print(f"Class Weights: {cw_dict}")
    
    model = build_model()
    
    # Callbacks with FULL LOGGING
    log_file = os.path.join(CONFIG['RESULTS_DIR'], "training_log.csv")
    model_file = os.path.join(CONFIG['RESULTS_DIR'], "best_model.h5")
    
    cb = [
        callbacks.ModelCheckpoint(model_file, save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.CSVLogger(log_file)  # SAVES EPOCH HISTORY
    ]
    
    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        callbacks=cb,
        class_weight=cw_dict
    )
    
    # Final Eval
    print("Evaluating...")
    best = models.load_model(model_file, custom_objects={'SpatialAttention': SpatialAttention})
    preds = (best.predict(X_test) > 0.5).astype(int)
    
    report = classification_report(y_test, preds, target_names=['Real (0)', 'Fake (1)'])
    conf_mat = confusion_matrix(y_test, preds)
    
    print(report)
    print("Confusion Matrix:\n", conf_mat)
    
    # Save Final Report to File
    report_file = os.path.join(CONFIG['RESULTS_DIR'], "final_report.txt")
    with open(report_file, "w") as f:
        f.write("FINAL CLASSIFICATION REPORT\n")
        f.write("===========================\n")
        f.write(report)
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("================\n")
        f.write(str(conf_mat))
    
    print(f"All results saved to {CONFIG['RESULTS_DIR']}")

if __name__ == "__main__":
    run_preprocessing()
    run_training()