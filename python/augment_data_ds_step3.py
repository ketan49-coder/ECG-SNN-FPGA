import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Augmentation for 131-Feature Dataset (128 Waveform + 3 RR Features)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data_ds_step3"
OUTPUT_DIR = "data_ds_step3"
TARGET_PER_MINORITY = 20000

def augment_class(X_class, n_target):
    n_samples, n_feat = X_class.shape
    n_needed = max(0, n_target - n_samples)
    if n_needed == 0: return X_class
    idx = np.random.choice(n_samples, size=n_needed, replace=True)
    aug = X_class[idx].copy()

    # 1. Waveform perturbations (first 128 features)
    wave = aug[:, :128]
    shifts = np.random.randint(-4, 5, size=n_needed)
    for s in np.unique(shifts):
        if s != 0: wave[shifts == s] = np.roll(wave[shifts == s], s, axis=1)
    wave += np.random.normal(0.0, 0.015, size=wave.shape).astype(np.float32)
    wave *= np.random.uniform(0.90, 1.10, size=(n_needed, 1)).astype(np.float32)
    lens = np.random.randint(4, 13, size=n_needed)
    starts = np.random.randint(0, 128 - 12, size=n_needed)
    for i in range(n_needed): wave[i, starts[i]:starts[i]+lens[i]] = 0.285 # Baseline restore
    aug[:, :128] = wave

    # 2. RR timing perturbations (last 3 features: subtle biological jitter +/- 1.5%)
    rr = aug[:, 128:]
    rr += np.random.normal(0.0, 0.015, size=rr.shape).astype(np.float32)
    aug[:, 128:] = rr
    
    return np.vstack([X_class, np.clip(aug, 0.0, 1.0)])

def main():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_bal, y_bal = [], []

    for c in range(4):
        mask = (y_train == c)
        Xc = X_train[mask]
        if c == 0:
            X_bal.append(Xc); y_bal.append(np.full(len(Xc), c, dtype=np.int64))
        else:
            Xc_aug = augment_class(Xc, TARGET_PER_MINORITY)
            X_bal.append(Xc_aug); y_bal.append(np.full(len(Xc_aug), c, dtype=np.int64))

    X_train_aug = np.vstack(X_bal)
    y_train_aug = np.concatenate(y_bal)
    perm = np.random.permutation(len(X_train_aug))
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_aug[perm])
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train_aug[perm])
    print(f"Balanced Step 3 DS1 Train Samples: {len(X_train_aug)} (Features: {X_train_aug.shape[1]})")

if __name__ == "__main__":
    main()
