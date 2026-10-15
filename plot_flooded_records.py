import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_PATH = r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"
OUTPUT_IMG = "waveform_inspection_flooded.png"

records_to_plot = ['117', '214', '231', '111', '232']
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=False)

for idx, rec in enumerate(records_to_plot):
    r = wfdb.rdrecord(os.path.join(DATA_PATH, rec), channels=[0], sampto=2000)
    sig = r.p_signal.flatten()
    ann = wfdb.rdann(os.path.join(DATA_PATH, rec), 'atr', sampto=2000)
    
    ax = axes[idx]
    ax.plot(sig, color='royalblue', lw=1.2, label=f"Record {rec} Raw Signal (Lead {r.sig_name[0]})")
    
    # Plot annotations
    for p, s in zip(ann.sample, ann.symbol):
        ax.axvline(p, color='red', linestyle='--', alpha=0.5)
        ax.text(p, np.max(sig)*0.85, s, color='crimson', fontweight='bold', fontsize=9)
        
    ax.set_title(f"Patient {rec} - Clinical Lead Profile", fontsize=11, fontweight='bold')
    ax.set_ylabel("mV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel("Sample Index (at 360 Hz)")
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150)
print(f"Successfully saved waveform plot to {OUTPUT_IMG}")
