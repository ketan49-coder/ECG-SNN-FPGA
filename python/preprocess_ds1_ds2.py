import wfdb
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# Path Configuration (Autodetects Colab Drive or Local Directory)
# ─────────────────────────────────────────────────────────────────────────────
POSSIBLE_PATHS = [
    r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0",
    r"mit-bih-arrhythmia-database-1.0.0",
    r"/content/drive/MyDrive/SNN-ECG/mit-bih-arrhythmia-database-1.0.0",
    r"/content/mit-bih-arrhythmia-database-1.0.0"
]

DATASET_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and (os.path.exists(os.path.join(p, "100.dat")) or os.path.exists(os.path.join(p, "100.hea"))):
        DATASET_DIR = p
        break

if DATASET_DIR is None:
    DATASET_DIR = r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

OUTPUT_DIR = "data_ds"
WINDOW_SIZE = 128  # 128 samples per heartbeat

# ─────────────────────────────────────────────────────────────────────────────
# 4-Class AAMI Mapping (De Chazal Standard: Q is dropped)
# ─────────────────────────────────────────────────────────────────────────────
# Class Q is dropped per standard benchmark protocol since paced records
# (102, 104, 107, 217) are excluded, leaving Q virtually empty (<0.02%).
AAMI_4CLASS_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'S': 1, 'J': 1,          # Supraventricular (S)
    'V': 2, 'E': 2,                          # Ventricular (V)
    'F': 3                                   # Fusion (F)
}
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']

# ─────────────────────────────────────────────────────────────────────────────
# Strict Patient Record Partitioning (De Chazal et al., IEEE TBME 2004)
# ─────────────────────────────────────────────────────────────────────────────
# 4 Paced records strictly dropped: 102, 104, 107, 217

# DS1 partitioned into Train (18 patients) and Val (4 patients) by RECORD
DS1_TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '230'
]

DS1_VAL_RECORDS = [
    '209', '215', '220', '223'  # 4 validation patients from DS1
]

# DS2 Test Set (22 patients) - Strictly untouched during training & tuning
DS2_TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

def extract_beats_from_records(record_list):
    X, y = [], []
    for record_name in record_list:
        record_path = os.path.join(DATASET_DIR, record_name)
        try:
            record = wfdb.rdrecord(record_path, channels=[0])
            signal = record.p_signal.flatten()
            annotation = wfdb.rdann(record_path, 'atr')
            peaks = annotation.sample
            symbols = annotation.symbol
            
            for peak, symbol in zip(peaks, symbols):
                if symbol in AAMI_4CLASS_MAPPING:
                    left = peak - (WINDOW_SIZE // 2)
                    right = peak + (WINDOW_SIZE // 2)
                    if left >= 0 and right < len(signal):
                        window = signal[left:right]
                        min_v, max_v = np.min(window), np.max(window)
                        if max_v > min_v:
                            window_norm = (window - min_v) / (max_v - min_v)
                            X.append(window_norm)
                            y.append(AAMI_4CLASS_MAPPING[symbol])
        except Exception as e:
            print(f"  Warning: could not read {record_name}: {e}")
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def print_distribution(y, set_name):
    print(f"\n{set_name} Class Distribution (Total: {len(y)} beats):")
    unique, counts = np.unique(y, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    for c in range(4):
        cnt = counts_dict.get(c, 0)
        pct = 100.0 * cnt / len(y) if len(y) > 0 else 0
        print(f"  Class {c} ({CLASS_NAMES[c]:<22}): {cnt:>6d} ({pct:>5.2f}%)")

def main():
    print("=" * 70)
    print("DE CHAZAL (2004) RIGOROUS INTER-PATIENT PREPROCESSING (4-CLASS)")
    print("=" * 70)
    print(f"Reading database from: {DATASET_DIR}")
    print("Excluded Paced Records: 102, 104, 107, 217 (per AAMI EC57 protocol)")
    print("Class Q: Excluded (presents < 0.02% of non-paced beats)")
    print(f"Partitions: DS1 Train (18 pts) | DS1 Val (4 pts) | DS2 Test (22 pts)\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Extracting DS1 Train (18 Records)...")
    X_train, y_train = extract_beats_from_records(DS1_TRAIN_RECORDS)
    print_distribution(y_train, "DS1 Train (18 Patients)")
    
    print("\n2. Extracting DS1 Val (4 Records)...")
    X_val, y_val = extract_beats_from_records(DS1_VAL_RECORDS)
    print_distribution(y_val, "DS1 Val (4 Patients)")
    
    print("\n3. Extracting DS2 Test (22 Records - COMPLETELY INDEPENDENT)...")
    X_test, y_test = extract_beats_from_records(DS2_TEST_RECORDS)
    print_distribution(y_test, "DS2 Test (22 Patients)")
    
    # Save partitioned data into data_ds/
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    
    print(f"\n=======================================================")
    print(f"SUCCESS: Saved Inter-Patient datasets to '{OUTPUT_DIR}/'")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Validation Leakage: ZERO (Record-level partition from DS1)")
    print(f"Test Snooping: ZERO (DS2 only touched at final evaluation)")
    print("=======================================================")

if __name__ == "__main__":
    main()
