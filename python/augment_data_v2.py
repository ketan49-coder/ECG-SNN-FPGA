import os
import numpy as np

DATA_DIR   = "data"
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

def add_gaussian_noise(windows, sigma=0.03):
    return np.clip(windows + np.random.normal(0, sigma, windows.shape), 0.0, 1.0)

def amplitude_scale(windows, scale_range=(0.80, 1.20)):
    scales = np.random.uniform(*scale_range, (windows.shape[0], 1))
    return np.clip(windows * scales, 0.0, 1.0)

def mixup(windows, alpha=0.3):
    n = len(windows)
    idx = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, (n, 1))
    return np.clip(lam * windows + (1 - lam) * windows[idx], 0.0, 1.0)

def random_cutout(windows, max_cut=20):
    # Vectorized cutout
    n, w_len = windows.shape
    out = windows.copy()
    cut_lens = np.random.randint(5, max_cut + 1, size=n)
    cut_starts = np.random.randint(0, w_len - max_cut, size=n)
    
    # Create a boolean mask for cutouts
    mask = np.zeros_like(windows, dtype=bool)
    for i in range(n):
        mask[i, cut_starts[i]:cut_starts[i]+cut_lens[i]] = True
    
    out[mask] = 0.0
    return out

def augment_batch(X):
    # Simplified to just the fastest/most effective augmentations
    X = add_gaussian_noise(X.copy())
    X = amplitude_scale(X)
    if np.random.rand() > 0.5: X = mixup(X)
    if np.random.rand() > 0.5: X = random_cutout(X)
    return X

def augment_dataset():
    print("Loading extracted dataset...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    target_count = 20000 
    print(f"\nTarget minority count capped at: {target_count}")

    X_aug = [X_train]
    y_aug = [y_train]

    for class_idx in range(1, 5): 
        mask    = (y_train == class_idx)
        X_class = X_train[mask]
        current = len(X_class)

        if current == 0: continue
        
        needed = target_count - current
        if needed <= 0: continue

        print(f"  {CLASS_NAMES[class_idx]}: {current} → {target_count} (+{needed} synthetic samples)")

        # Generate all needed samples at once
        idx = np.random.choice(current, needed, replace=True)
        X_batch = X_class[idx].copy()
        
        # Apply vectorized augmentation
        X_gen = augment_batch(X_batch)
        
        X_aug.append(X_gen)
        y_aug.append(np.full(needed, class_idx, dtype=np.int64))

    print("Concatenating and saving...")
    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    
    # Fast shuffle
    idx = np.random.permutation(len(X_out))
    X_out = X_out[idx]
    y_out = y_out[idx]

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_out.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_out.astype(np.int64))
    print(f"\nAugmented training size: {len(X_out)} samples. Saved to data/ ✅")

if __name__ == "__main__":
    augment_dataset()
