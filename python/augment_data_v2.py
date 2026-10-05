"""
augment_data_v2.py — Aggressive Augmentation for Minority Classes
==================================================================
Key improvements over augment_data.py:
  1. Target count = 80% of Normal class (not just largest minority)
     → All minority classes get much more data
  2. Mixup augmentation between same class samples
     → Creates smoother decision boundaries for rare classes
  3. Random cutout (zero out random segment)
     → Simulates lead noise / signal dropout in wearables
  4. Reports per-class balance ratio before and after
"""

import os
import numpy as np

DATA_DIR   = "data"
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)',
               'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']


# ─────────────────────────────────────────────
# Augmentation functions
# ─────────────────────────────────────────────

def add_gaussian_noise(windows, sigma=0.03):
    """Electrode / skin contact variation."""
    return np.clip(windows + np.random.normal(0, sigma, windows.shape), 0.0, 1.0)


def time_shift(windows, max_shift=10):
    """R-peak not perfectly centred — common in real recordings."""
    shifted = np.zeros_like(windows)
    for i, w in enumerate(windows):
        s = np.random.randint(-max_shift, max_shift + 1)
        if s > 0:   shifted[i, s:]  = w[:-s]
        elif s < 0: shifted[i, :s]  = w[-s:]
        else:       shifted[i]      = w
    return shifted


def amplitude_scale(windows, scale_range=(0.80, 1.20)):
    """Different electrode placement / patient-specific amplitude."""
    scales = np.random.uniform(*scale_range, (windows.shape[0], 1))
    return np.clip(windows * scales, 0.0, 1.0)


def baseline_wander(windows, max_wander=0.07):
    """Breathing-induced slow drift in the ECG baseline."""
    x = np.linspace(0, 2 * np.pi, windows.shape[1])
    out = windows.copy()
    for i in range(len(out)):
        freq = np.random.uniform(0.3, 2.5)
        amp  = np.random.uniform(0, max_wander)
        out[i] = np.clip(out[i] + amp * np.sin(freq * x + np.random.uniform(0, np.pi)), 0.0, 1.0)
    return out


def mixup(windows, alpha=0.3):
    """
    Mixup within same class: blend two samples with a random weight.
    Creates smoother intra-class variation without changing the label.
    """
    n   = len(windows)
    idx = np.random.permutation(n)
    lam = np.random.beta(alpha, alpha, (n, 1))
    return np.clip(lam * windows + (1 - lam) * windows[idx], 0.0, 1.0)


def random_cutout(windows, max_cut=20):
    """
    Zero out a random contiguous segment.
    Simulates lead noise / signal loss in a wearable sensor.
    """
    out = windows.copy()
    for i in range(len(out)):
        cut_len   = np.random.randint(5, max_cut + 1)
        cut_start = np.random.randint(0, windows.shape[1] - cut_len)
        out[i, cut_start:cut_start + cut_len] = 0.0
    return out


def augment_batch(X, method='all'):
    """Apply a random combination of augmentations."""
    X = add_gaussian_noise(X.copy())
    X = time_shift(X)
    X = amplitude_scale(X)
    X = baseline_wander(X)
    # 50% chance of applying mixup and cutout
    if np.random.rand() > 0.5:
        X = mixup(X)
    if np.random.rand() > 0.5:
        X = random_cutout(X)
    return X


# ─────────────────────────────────────────────
# Main augmentation pipeline
# ─────────────────────────────────────────────

def augment_dataset():
    print("Loading extracted dataset...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    print(f"Original training size: {len(X_train)} samples\n")
    unique, counts = np.unique(y_train, return_counts=True)

    print("Class distribution BEFORE augmentation:")
    for c, cnt in zip(unique, counts):
        bar = "█" * (cnt // 500)
        print(f"  {CLASS_NAMES[c]:<30} {cnt:6d}  {bar}")

    normal_count = counts[0]

    # Target: bring ALL minority classes to 80% of Normal count
    # This is much more aggressive than the old "match largest minority"
    target_count = int(normal_count * 0.80)
    print(f"\nNormal class count    : {normal_count}")
    print(f"Target minority count : {target_count} (80% of Normal)\n")

    X_aug = [X_train]
    y_aug = [y_train]

    for class_idx in range(1, 5):   # skip Normal (class 0)
        mask    = (y_train == class_idx)
        X_class = X_train[mask]
        current = len(X_class)

        if current == 0:
            print(f"  {CLASS_NAMES[class_idx]}: No samples — skipping.")
            continue

        needed = target_count - current
        if needed <= 0:
            print(f"  {CLASS_NAMES[class_idx]}: Already {current} samples — no augmentation needed.")
            continue

        print(f"  {CLASS_NAMES[class_idx]}: {current} → {target_count} (+{needed} synthetic samples)")

        # Generate in batches to apply diverse augmentation combos
        generated = []
        while len(generated) < needed:
            batch_size = min(current, needed - len(generated))
            idx        = np.random.choice(current, batch_size, replace=True)
            X_batch    = X_class[idx].copy()
            X_batch    = augment_batch(X_batch)
            generated.append(X_batch)

        X_gen = np.concatenate(generated, axis=0)[:needed]
        X_aug.append(X_gen)
        y_aug.append(np.full(needed, class_idx, dtype=np.int64))

    # Concatenate and shuffle
    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    idx   = np.random.permutation(len(X_out))
    X_out, y_out = X_out[idx], y_out[idx]

    print(f"\nAugmented training size: {len(X_out)} samples")
    print("\nClass distribution AFTER augmentation:")
    unique2, counts2 = np.unique(y_out, return_counts=True)
    for c, cnt in zip(unique2, counts2):
        ratio = cnt / counts2[0]
        bar   = "█" * (cnt // 500)
        print(f"  {CLASS_NAMES[c]:<30} {cnt:6d}  ratio={ratio:.2f}  {bar}")

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_out.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_out.astype(np.int64))
    print("\nSaved to data/ — run training now!")


if __name__ == "__main__":
    augment_dataset()
