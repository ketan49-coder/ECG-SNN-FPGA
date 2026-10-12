import os
import wfdb
import numpy as np
import scipy.signal

AAMI = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
        'A': 1, 'a': 1, 'S': 1, 'J': 1,
        'V': 2, 'E': 2,
        'F': 3}

DS1_TRAIN = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '122', '201', '203', '207', '208', '230']
OUTPUT_DIR = "data_ds1_svdb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Locate directories
mit_dir = [r for r, _, f in os.walk('/content', followlinks=True) if '100.dat' in f][0]
svdb_dir = [r for r, _, f in os.walk('/content', followlinks=True) if '800.dat' in f][0]
print(f"MIT-BIH Directory: {mit_dir}")
print(f"SVDB Directory:    {svdb_dir}")

def extract_record_step3(rec_path, is_svdb=False):
    X, y = [], []
    try:
        r = wfdb.rdrecord(rec_path, channels=[0])
        sig = r.p_signal.flatten()
        ann = wfdb.rdann(rec_path, 'atr')
        
        # Resample SVDB from 128 Hz to 360 Hz
        if is_svdb:
            sig = scipy.signal.resample_poly(sig, 45, 16)
            samples = np.round(ann.sample * 45 / 16).astype(int)
        else:
            samples = ann.sample
            
        symbols = ann.symbol
        n_beats = len(samples)
        
        # Robust per-record normalization
        med = np.median(sig)
        mad = np.median(np.abs(sig - med)) * 1.4826 + 1e-4
        sig_norm = np.clip(((sig - med) / mad + 4.0) / 14.0, 0.0, 1.0)
        
        history_rrs = []
        for i in range(n_beats):
            p, s = samples[i], symbols[i]
            if s not in AAMI or (p - 64 < 0) or (p + 64 >= len(sig_norm)):
                continue
                
            rr_pre = (samples[i] - samples[i-1]) / 360.0 if i > 0 else (samples[1] - samples[0]) / 360.0
            rr_post = (samples[i+1] - samples[i]) / 360.0 if i < n_beats - 1 else rr_pre
            
            history_rrs.append(rr_pre)
            if len(history_rrs) > 10:
                history_rrs.pop(0)
            local_mean = float(np.mean(history_rrs))
            rr_ratio = rr_pre / (local_mean + 1e-4)
            
            feat_pre = np.clip(rr_pre / 2.0, 0.0, 1.0)
            feat_post = np.clip(rr_post / 2.0, 0.0, 1.0)
            feat_ratio = np.clip(rr_ratio / 2.0, 0.0, 1.0)
            
            vec = np.hstack([sig_norm[p-64 : p+64], [feat_pre, feat_post, feat_ratio]])
            X.append(vec)
            y.append(AAMI[s])
    except Exception as e:
        print(f"Error {rec_path}: {e}")
    return X, y

print("\n--- Extracting DS1 Training Records (15 patients) ---")
X_all, y_all = [], []
for rec in DS1_TRAIN:
    x_r, y_r = extract_record_step3(os.path.join(mit_dir, rec), is_svdb=False)
    X_all.extend(x_r); y_all.extend(y_r)
print(f"Extracted {len(X_all)} beats from DS1 Train.")

print("\n--- Extracting SVDB Records (78 patients, Channel 0 resampled 128->360 Hz) ---")
svdb_recs = sorted(list(set([f.split('.')[0] for f in os.listdir(svdb_dir) if f.startswith('8')])))
for rec in svdb_recs:
    x_r, y_r = extract_record_step3(os.path.join(svdb_dir, rec), is_svdb=True)
    X_all.extend(x_r); y_all.extend(y_r)

X_all, y_all = np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int64)
print(f"Total Raw Combined Training Beats: {len(X_all)}")
counts = {c: np.sum(y_all == c) for c in range(4)}
print(f"Raw Class Breakdown: N={counts[0]}, S={counts[1]}, V={counts[2]}, F={counts[3]}")

# Balance combined training set to 25,000 per class
def augment_class(Xc, n_target=25000):
    n = len(Xc)
    if n >= n_target:
        idx = np.random.choice(n, size=n_target, replace=False)
        return Xc[idx]
    n_needed = n_target - n
    idx = np.random.choice(n, size=n_needed, replace=True)
    aug = Xc[idx].copy()
    
    # Waveform jitter
    wave = aug[:, :128]
    shifts = np.random.randint(-4, 5, size=n_needed)
    for s in np.unique(shifts):
        if s != 0: wave[shifts == s] = np.roll(wave[shifts == s], s, axis=1)
    wave += np.random.normal(0.0, 0.015, size=wave.shape).astype(np.float32)
    wave *= np.random.uniform(0.90, 1.10, size=(n_needed, 1)).astype(np.float32)
    aug[:, :128] = wave
    
    # RR timing jitter
    aug[:, 128:] += np.random.normal(0.0, 0.015, size=aug[:, 128:].shape).astype(np.float32)
    return np.vstack([Xc, np.clip(aug, 0.0, 1.0)])

print("\nBalancing classes to 25,000 beats each...")
X_bal, y_bal = [], []
for c in range(4):
    Xc = X_all[y_all == c]
    Xc_b = augment_class(Xc, n_target=25000)
    X_bal.append(Xc_b)
    y_bal.append(np.full(len(Xc_b), c, dtype=np.int64))

X_train_bal = np.vstack(X_bal)
y_train_bal = np.concatenate(y_bal)
p = np.random.permutation(len(X_train_bal))
X_train_bal, y_train_bal = X_train_bal[p], y_train_bal[p]

np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_bal)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train_bal)

# Copy validation and test directly from data_ds_step3 to guarantee identical benchmarks
for f_name in ['X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']:
    src = os.path.join("data_ds_step3", f_name)
    dst = os.path.join(OUTPUT_DIR, f_name)
    if os.path.exists(src):
        np.save(dst, np.load(src))

print(f"\n--> SUCCESS: Balanced DS1+SVDB Training set ready: {X_train_bal.shape}")
print(f"Saved into '{OUTPUT_DIR}/'")
