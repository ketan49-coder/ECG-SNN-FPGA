import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fast Vectorized Augmentation for DS1 Train Set (4-Class)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data_ds"
OUTPUT_DIR = "data_ds"
TARGET_PER_MINORITY = 20000  # Target count for S, V, F classes

def augment_class_vectorized(X_class, n_target):
    n_samples, win_len = X_class.shape
    if n_samples == 0:
        return X_class
    n_needed = max(0, n_target - n_samples)
    if n_needed == 0:
        return X_class

    indices = np.random.choice(n_samples, size=n_needed, replace=True)
    aug = X_class[indices].copy()

    # 1. Temporal jitter / roll: -4 to +4 samples
    shifts = np.random.randint(-4, 5, size=n_needed)
    for shift_val in np.unique(shifts):
        if shift_val != 0:
            mask = (shifts == shift_val)
            aug[mask] = np.roll(aug[mask], shift_val, axis=1)

    # 2. Gaussian Noise
    noise = np.random.normal(0.0, 0.02, size=aug.shape).astype(np.float32)
    aug += noise

    # 3. Amplitude Scaling
    scales = np.random.uniform(0.85, 1.15, size=(n_needed, 1)).astype(np.float32)
    aug *= scales

    # 4. Random Cutout
    cut_lens = np.random.randint(4, 13, size=n_needed)
    cut_starts = np.random.randint(0, win_len - 12, size=n_needed)
    for i in range(n_needed):
        aug[i, cut_starts[i] : cut_starts[i] + cut_lens[i]] = 0.0

    # 5. Dynamic range normalization [0, 1]
    mins = aug.min(axis=1, keepdims=True)
    maxs = aug.max(axis=1, keepdims=True)
    denom = np.where((maxs - mins) > 1e-6, maxs - mins, 1.0)
    aug = (aug - mins) / denom

    return np.vstack([X_class, aug])

def main():
    print("=" * 65)
    print("AUGMENTING DS1 INTER-PATIENT TRAINING SET (4-CLASS: N, S, V, F)")
    print("=" * 65)
    
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    
    print(f"Original DS1 Train Shape: {X_train.shape}")
    
    X_balanced = []
    y_balanced = []
    
    for c in range(4):
        mask = (y_train == c)
        X_c = X_train[mask]
        n_c = len(X_c)
        
        if c == 0:
            # Keep all Normal beats
            X_balanced.append(X_c)
            y_balanced.append(np.full(n_c, c, dtype=np.int64))
            print(f"  Class 0 (Normal): kept {n_c} samples as-is.")
        else:
            # Augment minority classes S, V, F to TARGET_PER_MINORITY
            X_c_aug = augment_class_vectorized(X_c, TARGET_PER_MINORITY)
            X_balanced.append(X_c_aug)
            y_balanced.append(np.full(len(X_c_aug), c, dtype=np.int64))
            print(f"  Class {c}: {n_c} -> augmented to {len(X_c_aug)} samples.")
                
    X_train_aug = np.vstack(X_balanced)
    y_train_aug = np.concatenate(y_balanced)
    
    # Shuffle
    perm = np.random.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[perm]
    y_train_aug = y_train_aug[perm]
    
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_aug)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train_aug)
    
    print("\nBalanced DS1 Training Set:")
    print(f"  Total samples: {len(X_train_aug)}")
    print(f"  Saved to '{OUTPUT_DIR}/X_train.npy' and 'y_train.npy'")
    print("=" * 65)

if __name__ == "__main__":
    main()
