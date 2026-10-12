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

DS1_TRAIN = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '122', '201', '203', '207', '208', '230']
DS1_VAL   = ['119', '124', '205', '209', '215', '220', '223']
DS2_TEST  = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
TARGET_OUTLIERS = ['117', '214', '231', '111']

def compute_record_stats(rec):
    r = wfdb.rdrecord(os.path.join(DATASET_DIR, rec), channels=[0])
    sig = r.p_signal.flatten()
    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) * 1.4826  # Normalized MAD
    q25, q75 = np.percentile(sig, [25, 75])
    iqr = q75 - q25
    peak_r = np.percentile(sig, 99.5) # Approximate R-peak height
    min_v = np.percentile(sig, 0.5)
    dyn_range = peak_r - min_v
    return {
        'rec': rec,
        'median': med,
        'mad': mad,
        'iqr': iqr,
        'peak_r': peak_r,
        'dyn_range': dyn_range
    }

def main():
    print("=" * 85)
    print("STATISTICAL AUDIT OF PATIENT BASELINE, SCALE, AND DYNAMIC RANGE")
    print("=" * 85)
    
    train_stats = [compute_record_stats(r) for r in DS1_TRAIN]
    val_stats   = [compute_record_stats(r) for r in DS1_VAL]
    test_stats  = [compute_record_stats(r) for r in DS2_TEST]
    
    tr_mads = [s['mad'] for s in train_stats]
    tr_medians = [s['median'] for s in train_stats]
    tr_dyn = [s['dyn_range'] for s in train_stats]
    
    print(f"DS1 Training Set Baseline Statistics (15 Patients):")
    print(f"  Signal Median : mean={np.mean(tr_medians):.4f}, min={np.min(tr_medians):.4f}, max={np.max(tr_medians):.4f}")
    print(f"  MAD (Scale)   : mean={np.mean(tr_mads):.4f}, min={np.min(tr_mads):.4f}, max={np.max(tr_mads):.4f}")
    print(f"  Dynamic Range : mean={np.mean(tr_dyn):.4f}, min={np.min(tr_dyn):.4f}, max={np.max(tr_dyn):.4f}")
    
    print("\n" + "-" * 85)
    print(f"{'Group':<10} | {'Record':<6} | {'Median':<10} | {'MAD (Scale)':<12} | {'IQR':<10} | {'Dynamic Range':<14} | Notes")
    print("-" * 85)
    
    for s in test_stats:
        flag = ""
        if s['rec'] in TARGET_OUTLIERS:
            flag = " <-- PRIMARY FALSE-S SOURCE"
        elif s['rec'] in ['232', '222']:
            flag = " <-- TRUE-S SOURCE (75% of S)"
            
        is_mad_outlier = s['mad'] > np.max(tr_mads) or s['mad'] < np.min(tr_mads)
        is_med_outlier = s['median'] > np.max(tr_medians) or s['median'] < np.min(tr_medians)
        
        note = flag
        if is_mad_outlier and flag == "": note = " [Scale Outlier]"
        if is_med_outlier and flag == "": note += " [Median Outlier]"
        
        print(f"DS2 Test   | {s['rec']:<6} | {s['median']:>9.4f} | {s['mad']:>11.4f} | {s['iqr']:>9.4f} | {s['dyn_range']:>13.4f} |{note}")

    print("=" * 85)

if __name__ == "__main__":
    main()
