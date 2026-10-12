import wfdb
import numpy as np
import os

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

OUTPUT_DIR = "data_ds"
WINDOW_SIZE = 128

AAMI_4CLASS_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'S': 1, 'J': 1,
    'V': 2, 'E': 2,
    'F': 3
}
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']

# STEP 1: Increase validation size to 7 patients for a stable, low-noise signal
DS1_TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '122',
    '201', '203', '207', '208', '230'
] # 15 training patients

DS1_VAL_RECORDS = [
    '119', '124', '205', '209', '215', '220', '223'
] # 7 validation patients (broad clinical diversity)

DS2_TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
] # 22 test patients (strictly untouched)

def extract_beats(records):
    X, y = [], []
    for rec in records:
        try:
            r = wfdb.rdrecord(os.path.join(DATASET_DIR, rec), channels=[0])
            sig = r.p_signal.flatten()
            ann = wfdb.rdann(os.path.join(DATASET_DIR, rec), 'atr')
            for p, s in zip(ann.sample, ann.symbol):
                if s in AAMI_4CLASS_MAPPING:
                    left, right = p - 64, p + 64
                    if left >= 0 and right < len(sig):
                        w = sig[left:right]
                        mi, ma = np.min(w), np.max(w)
                        if ma > mi:
                            X.append((w - mi) / (ma - mi))
                            y.append(AAMI_4CLASS_MAPPING[s])
        except Exception as e:
            print(f"Error reading {rec}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Extracting DS1 Train (15 patients)...")
    X_train, y_train = extract_beats(DS1_TRAIN_RECORDS)
    print("Extracting DS1 Val (7 patients)...")
    X_val, y_val = extract_beats(DS1_VAL_RECORDS)
    print("Extracting DS2 Test (22 patients)...")
    X_test, y_test = extract_beats(DS2_TEST_RECORDS)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)
    print(f"Done! Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    main()
