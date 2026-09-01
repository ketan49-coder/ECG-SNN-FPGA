import os
import numpy as np

DATA_DIR = "data"

# Class names for reference
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

def add_gaussian_noise(windows, sigma=0.02):
    """Add small random Gaussian noise to simulate electrode/skin variation."""
    noise = np.random.normal(0, sigma, windows.shape)
    return np.clip(windows + noise, 0.0, 1.0)

def time_shift(windows, max_shift=8):
    """
    Slightly shift the heartbeat window left or right.
    Simulates the R-peak not being perfectly centered (which happens in real recordings).
    """
    shifted = np.zeros_like(windows)
    for i, w in enumerate(windows):
        shift = np.random.randint(-max_shift, max_shift)
        if shift > 0:
            shifted[i, shift:] = w[:-shift]
        elif shift < 0:
            shifted[i, :shift] = w[-shift:]
        else:
            shifted[i] = w
    return shifted

def amplitude_scale(windows, scale_range=(0.85, 1.15)):
    """
    Randomly scale the amplitude of the heartbeat.
    Simulates different electrode placements and patient-specific amplitudes.
    """
    scales = np.random.uniform(scale_range[0], scale_range[1], (windows.shape[0], 1))
    return np.clip(windows * scales, 0.0, 1.0)

def baseline_wander(windows, max_wander=0.05):
    """
    Add a slow sinusoidal drift to the baseline.
    This simulates the breathing-induced baseline wander seen in real ECGs.
    """
    x = np.linspace(0, 2 * np.pi, windows.shape[1])
    for i in range(len(windows)):
        freq      = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(0, max_wander)
        wander    = amplitude * np.sin(freq * x)
        windows[i] = np.clip(windows[i] + wander, 0.0, 1.0)
    return windows

def augment_dataset():
    print("Loading extracted dataset...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    print(f"Original training size: {len(X_train)} samples\n")
    print("Class distribution before augmentation:")
    unique, counts = np.unique(y_train, return_counts=True)
    for c, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[c]}: {count}")

    # Target: make all minority classes match the largest minority class
    # We deliberately do NOT oversample Normal to avoid making the imbalance worse
    target_count = max(counts[1:])  # Largest count among minority classes (ignore Normal)
    
    print(f"\nTarget count for each minority class: {target_count}")
    
    X_aug = [X_train]
    y_aug = [y_train]
    
    # Apply augmentation only to minority classes (S, V, F, Q) — skip Normal (class 0)
    for class_idx in range(1, 5):
        class_mask    = y_train == class_idx
        X_class       = X_train[class_mask]
        current_count = len(X_class)
        
        if current_count == 0:
            print(f"  {CLASS_NAMES[class_idx]}: No samples found, skipping.")
            continue
        
        needed = target_count - current_count
        if needed <= 0:
            print(f"  {CLASS_NAMES[class_idx]}: Already has {current_count} samples, no augmentation needed.")
            continue
        
        print(f"  {CLASS_NAMES[class_idx]}: Generating {needed} synthetic samples...")
        
        # Randomly pick existing samples to augment
        indices     = np.random.choice(current_count, needed, replace=True)
        X_to_aug    = X_class[indices].copy()
        
        # Apply all four augmentation techniques simultaneously
        X_noisy     = add_gaussian_noise(X_to_aug.copy())
        X_shifted   = time_shift(X_noisy)
        X_scaled    = amplitude_scale(X_shifted)
        X_wandered  = baseline_wander(X_scaled)
        
        X_aug.append(X_wandered)
        y_aug.append(np.full(needed, class_idx, dtype=np.int64))
    
    # Concatenate and shuffle
    X_train_aug = np.concatenate(X_aug, axis=0)
    y_train_aug = np.concatenate(y_aug, axis=0)
    
    shuffle_idx     = np.random.permutation(len(X_train_aug))
    X_train_aug     = X_train_aug[shuffle_idx]
    y_train_aug     = y_train_aug[shuffle_idx]
    
    print(f"\nAugmented training size: {len(X_train_aug)} samples")
    print("Class distribution after augmentation:")
    unique2, counts2 = np.unique(y_train_aug, return_counts=True)
    for c, count in zip(unique2, counts2):
        print(f"  {CLASS_NAMES[c]}: {count}")
    
    # Save augmented data (overwrite original training files)
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_aug.astype(np.float32))
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train_aug)
    
    print("\nAugmented data saved to 'data/' folder.")
    print("You can now run train_snn.py directly on the augmented dataset!")

if __name__ == "__main__":
    augment_dataset()
