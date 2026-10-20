"""
compare_s_beats.py
Diagnostic Script: Compares PyTorch Float32 vs. FPGA Integer Model on Supraventricular (S) beats.
Displays the exact spike counts for [N, S, V, F] to see if S is losing to N by just 1 or 2 spikes.
"""

import os
import torch
import numpy as np
from step2_reference_model import run_snn_integer_inference, TIMESTEPS, W1, W2, W3

# 1. Load PyTorch float model
CKPT_PATH = "checkpoints/lighter_smote_seed999_best.pth"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = "checkpoints/lighter_smote_seed42_best.pth"

from app import ECG_Spartan7_SNN
net = ECG_Spartan7_SNN()
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)
net.load_state_dict(state_dict, strict=False)
net.eval()

# 2. Load test set and isolate S beats
data = np.load("sparrow_test_set.npz")
X_test, y_test = data['X_test'], data['y_test']

# Find all S beat indices (label == 1)
s_indices = np.where(y_test == 1)[0]

print("=" * 80)
print(f"DIAGNOSTIC AUDIT: SUPRAVENTRICULAR (CLASS S) BEATS")
print(f"Total True S beats in test set: {len(s_indices)}")
print("=" * 80)
print(f"{'Beat #':<8} | {'PyTorch Float Pred':<18} {'PyTorch Spikes':<16} | {'FPGA Int Pred':<14} {'FPGA Spikes [N,S,V,F]':<22} | {'Margin (S - N)':<14}")
print("-" * 80)

classes = ['N', 'S', 'V', 'F']
s_wins = 0
tight_losses = 0

# Check first 20 S beats
for idx in s_indices[:20]:
    beat = X_test[idx]
    
    # PyTorch Float Inference
    with torch.no_grad():
        t_in = torch.tensor(beat, dtype=torch.float32).unsqueeze(0)
        spk_rec_py = net(t_in)
        py_counts = spk_rec_py.sum(dim=0).squeeze(0).numpy().astype(int)
        py_pred = classes[int(np.argmax(py_counts))]
        
    # FPGA Integer Inference
    int_pred_idx, int_counts, _ = run_snn_integer_inference(beat)
    int_pred = classes[int_pred_idx]
    
    diff_s_n = int_counts[1] - int_counts[0]
    
    if int_pred == 'S':
        s_wins += 1
        verdict = f"+{diff_s_n:2d} (S Wins ✅)"
    else:
        if abs(diff_s_n) <= 3:
            tight_losses += 1
            verdict = f"{diff_s_n:2d} (Tight loss ⚠️)"
        else:
            verdict = f"{diff_s_n:2d} (N Wins ❌)"
            
    print(f"#{idx:<7d} | {py_pred:<18} {str(list(py_counts)):<16} | {int_pred:<14} {str(list(int_counts)):<22} | {verdict}")

print("-" * 80)
print(f"Summary on first 20 S beats:")
print(f"  - FPGA S Detected cleanly : {s_wins} / 20")
print(f"  - Lost by only 1-3 spikes : {tight_losses} / 20 (Neuron S was right there!)")
print("=" * 80)
