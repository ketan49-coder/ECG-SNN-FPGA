import wfdb
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Preprocessing with Per-Record Robust Normalization (Median / MAD)
# ─────────────────────────────────────────────────────────────────────────────
POSSIBLE_PATHS = [
    r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0",
    r"mit-bih-arrhythmia-database-1.0.0",
    r"/content/drive/MyDrive/SNN-ECG/mit-bih-arrhythmia-database-1.0.0",
    r"/content/mit-bih-arrhythmia-database-1.0.0"
]

DATASET_DIR = None
for root, dirs, files in os.walk('/content'):
    if '100.hea' in files or '100.dat' in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    for p in POSSIBLE_PATHS:
        if os.path.exists(p) and (os.path.exists(os.path.join(p, "100.dat")) or os.path.exists(os.path.join(p, "100.hea"))):
            DATASET_DIR = p
            break

if DATASET_DIR is None:
    DATASET_DIR = r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

OUTPUT_DIR = "data_ds_step2"
WINDOW_SIZE = 128

AAMI_4CLASS_MAPPING = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'S': 1, 'J': 1, 'V': 2, 'E': 2, 'F': 3}

DS1_TRAIN_RECORDS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '122', '201', '203', '207', '208', '230']
DS1_VAL_RECORDS   = ['119', '124', '205', '209', '215', '220', '223']
DS2_TEST_RECORDS  = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

def extract_beats_robust_norm(records):
    X, y = [], []
    for rec in records:
        try:
            r = wfdb.rdrecord(os.path.join(DATASET_DIR, rec), channels=[0])
            sig = r.p_signal.flatten()
            ann = wfdb.rdann(os.path.join(DATASET_DIR, rec), 'atr')
            
            # STEP 2 CORE: Per-record robust centering and scaling
            med = np.median(sig)
            mad = np.median(np.abs(sig - med)) * 1.4826
            if mad < 1e-4: mad = np.std(sig) + 1e-4
            
            # Robust standardized signal (centers baseline at 0.0)
            sig_std = (sig - med) / mad
            
            # Fixed global linear map to [0, 1] across a [-4*MAD, +10*MAD] window
            # Baseline is locked at ~0.285 across ALL patients
            sig_norm = np.clip((sig_std + 4.0) / 14.0, 0.0, 1.0)
            
            for p, s in zip(ann.sample, ann.symbol):
                if s in AAMI_4CLASS_MAPPING and (p - 64 >= 0) and (p + 64 < len(sig)):
                    window = sig_norm[p - 64 : p + 64]
                    X.append(window)
                    y.append(AAMI_4CLASS_MAPPING[s])
        except Exception as e:
            print(f"Error {rec}: {e}")
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("=" * 80)
    print("STEP 2: PREPROCESSING WITH PER-RECORD ROBUST NORMALIZATION (MEDIAN/MAD)")
    print("=" * 80)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting 15 Train Patients (Per-Record MAD Normalized)...")
    X_tr, y_tr = extract_beats_robust_norm(DS1_TRAIN_RECORDS)
    print("Extracting 7 Val Patients (Per-Record MAD Normalized)...")
    X_va, y_va = extract_beats_robust_norm(DS1_VAL_RECORDS)
    print("Extracting 22 Test Patients (Per-Record MAD Normalized)...")
    X_te, y_te = extract_beats_robust_norm(DS2_TEST_RECORDS)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_tr); np.save(f"{OUTPUT_DIR}/y_train.npy", y_tr)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_va);   np.save(f"{OUTPUT_DIR}/y_val.npy", y_va)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_te);  np.save(f"{OUTPUT_DIR}/y_test.npy", y_te)
    
    print(f"\nDone! Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"Saved into '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
