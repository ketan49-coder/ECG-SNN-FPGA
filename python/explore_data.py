import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUTPUT_DIR = "../docs"

# Class names mapping
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

def explore_data():
    print("Loading data for exploration...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    
    print(f"Total training samples: {len(X_train)}")
    
    # 1. Plot Class Distribution
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar([CLASS_NAMES[i] for i in unique], counts, color='skyblue')
    plt.title('MIT-BIH Class Distribution (Training Set)')
    plt.ylabel('Number of Heartbeats')
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    print("Saved class_distribution.png")
    
    # 2. Plot one example of each class
    plt.figure(figsize=(15, 10))
    for i in range(5):
        # Find the first instance of class i
        idx = np.where(y_train == i)[0]
        if len(idx) > 0:
            first_idx = idx[0]
            sample = X_train[first_idx]
            
            plt.subplot(3, 2, i+1)
            plt.plot(sample, color='red' if i != 0 else 'blue')
            plt.title(f"Class: {CLASS_NAMES[i]}")
            plt.xlabel("Sample Index (0-127)")
            plt.ylabel("Normalized Amplitude")
            plt.grid(True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_waveforms.png"))
    print("Saved class_waveforms.png")

if __name__ == "__main__":
    explore_data()
