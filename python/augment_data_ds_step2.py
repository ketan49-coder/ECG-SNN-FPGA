import os
import numpy as np

DATA_DIR = "data_ds_step2"
OUTPUT_DIR = "data_ds_step2"
TARGET_PER_MINORITY = 20000

def augment_class(X_class, n_target):
    n_samples, win_len = X_class.shape
    n_needed = max(0, n_target - n_samples)
    if n_needed == 0: return X_class
    idx = np.random.choice(n_samples, size=n_needed, replace=True)
    aug = X_class[idx].copy()

    # Jitter, Noise, Scale, Cutout (preserving robust normalization dynamic range)
    shifts = np.random.randint(-4, 5, size=n_needed)
    for s in np.unique(shifts):
        if s != 0: aug[shifts == s] = np.roll(aug[shifts == s], s, axis=1)
    aug += np.random.normal(0.0, 0.015, size=aug.shape).astype(np.float32)
    aug *= np.random.uniform(0.90, 1.10, size=(n_needed, 1)).astype(np.float32)
    lens = np.random.randint(4, 13, size=n_needed)
    starts = np.random.randint(0, win_len - 12, size=n_needed)
    for i in range(n_needed): aug[i, starts[i]:starts[i]+lens[i]] = 0.285 # Set cutout to baseline
    
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
    print(f"Balanced Step 2 DS1 Train Samples: {len(X_train_aug)}")

if __name__ == "__main__":
    main()
