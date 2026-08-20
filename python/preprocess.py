import wfdb
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configure paths
# Update this path to where your dataset is extracted
DATASET_DIR = r"..\mit-bih-arrhythmia-database-1.0.0-20260819T152237Z-1-001\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"
OUTPUT_DIR = "data"

# Standard AAMI Classes mapping from MIT-BIH annotations
# N: Normal beat
# S: Supraventricular ectopic beat
# V: Ventricular ectopic beat
# F: Fusion beat
# Q: Unknown beat
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'S': 1, 'J': 1,          # Supraventricular (S)
    'V': 2, 'E': 2,                          # Ventricular (V)
    'F': 3,                                  # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                   # Unknown (Q)
}

# The list of 48 records in the MIT-BIH dataset
RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

WINDOW_SIZE = 128  # 128 samples per heartbeat

def preprocess_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    X_all = []
    y_all = []

    print(f"Reading records from: {DATASET_DIR}")
    for record_name in RECORDS:
        record_path = os.path.join(DATASET_DIR, record_name)
        
        try:
            # Read signal (we use channel 0: MLII)
            record = wfdb.rdrecord(record_path, channels=[0])
            signal = record.p_signal.flatten()
            
            # Read annotations (R-peaks and beat types)
            annotation = wfdb.rdann(record_path, 'atr')
            peaks = annotation.sample
            symbols = annotation.symbol
            
            for peak, symbol in zip(peaks, symbols):
                if symbol in AAMI_MAPPING:
                    # Check if window is within bounds
                    left = peak - (WINDOW_SIZE // 2)
                    right = peak + (WINDOW_SIZE // 2)
                    
                    if left >= 0 and right < len(signal):
                        window = signal[left:right]
                        
                        # Normalize window to [0, 1] for SNN rate coding
                        min_val = np.min(window)
                        max_val = np.max(window)
                        if max_val > min_val:
                            window_norm = (window - min_val) / (max_val - min_val)
                            
                            X_all.append(window_norm)
                            y_all.append(AAMI_MAPPING[symbol])
                            
        except Exception as e:
            print(f"Error reading record {record_name}: {e}")

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    print(f"\nExtracted {len(X_all)} total heartbeats.")
    print("Class distribution:")
    unique, counts = np.unique(y_all, return_counts=True)
    class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
    for c, count in zip(unique, counts):
        print(f"  {class_names[c]}: {count}")

    # Split into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    # Save as numpy arrays
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    
    print("\nData preprocessing complete. Saved to 'data/' folder.")

if __name__ == "__main__":
    preprocess_dataset()
