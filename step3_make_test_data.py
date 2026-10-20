"""
step3_make_test_data.py
Step 3: Generate 100 test heartbeat files (balanced across all 4 cardiac classes).
Distribution:
  - 40 Normal (N)
  - 20 Supraventricular (S)
  - 25 Ventricular (V)
  - 15 Fusion (F)
  Total = 100 Test Vectors

Saves:
  - Input spikes: 128 bits per line x 25 timesteps (for VHDL textio)
  - Raw 8-bit samples: 128 values (0..255) for data_in port
  - Expected output spikes: 4 bits per line x 25 timesteps
  - Expected final predicted class & spike counts
  - Master index file: test_vectors/test_manifest.txt
"""

import os
import numpy as np
from step2_reference_model import run_snn_integer_inference, TIMESTEPS

DATA_PATH = "sparrow_test_set.npz"
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] '{DATA_PATH}' not found!")
    exit(1)

data = np.load(DATA_PATH)
X_test, y_test = data['X_test'], data['y_test']

DIR_INPUTS = "test_vectors/inputs"
DIR_EXPECTED = "test_vectors/expected"
os.makedirs(DIR_INPUTS, exist_ok=True)
os.makedirs(DIR_EXPECTED, exist_ok=True)

class_names = ["N", "S", "V", "F"]
class_full   = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)"]

# Select stratified indices from test set
idx_N = np.where(y_test == 0)[0][:40]
idx_S = np.where(y_test == 1)[0][:20]
idx_V = np.where(y_test == 2)[0][:25]
idx_F = np.where(y_test == 3)[0][:15]

all_indices = list(idx_N) + list(idx_S) + list(idx_V) + list(idx_F)
total_cases = len(all_indices)

print("=" * 80)
print(f"STEP 3: GENERATING {total_cases} TEST VECTOR FILES FOR VIVADO SIMULATION")
print("=" * 80)
print(f"Stratified Distribution: 40 N | 20 S | 25 V | 15 F = {total_cases} Total Vectors")
print(f"Output directories:\n  - Inputs  : {DIR_INPUTS}/\n  - Expected: {DIR_EXPECTED}/\n")

np.random.seed(42)

manifest_lines = []
manifest_lines.append(f"{'Case':<6} {'Index':<8} {'True':<6} {'Pred':<6} {'Result':<10} {'Spike Counts [N, S, V, F]':<26} {'Input File'}\n")
manifest_lines.append("-" * 95 + "\n")

correct_per_class = {0: 0, 1: 0, 2: 0, 3: 0}
total_per_class   = {0: 0, 1: 0, 2: 0, 3: 0}

for case_num, idx in enumerate(all_indices):
    beat = X_test[idx]  # 128 normalized voltage values in [0, 1]
    true_label = y_test[idx]
    total_per_class[true_label] += 1
    
    # Run bit-true integer inference
    pred_class, spike_counts, out_spikes = run_snn_integer_inference(beat)
    
    is_match = (pred_class == true_label)
    if is_match:
        correct_per_class[true_label] += 1
        
    tag = f"beat_{case_num:03d}_{class_names[true_label]}"
    input_spike_file = os.path.join(DIR_INPUTS, f"{tag}_spikes.txt")
    input_sample_file = os.path.join(DIR_INPUTS, f"{tag}_samples.txt")
    expected_file = os.path.join(DIR_EXPECTED, f"{tag}_expected.txt")
    
    # 1. Write 25 timesteps of 128-bit input spikes (for textio)
    rand_matrix = np.random.rand(TIMESTEPS, 128)
    spikes_25x128 = (rand_matrix < beat).astype(np.int32)
    with open(input_spike_file, "w") as f_in:
        for t in range(TIMESTEPS):
            f_in.write("".join(str(b) for b in spikes_25x128[t]) + "\n")
            
    # 2. Write 128 raw 8-bit integer samples (0..255) for data_in port
    samples_8bit = np.clip(np.round(beat * 255.0), 0, 255).astype(np.uint8)
    with open(input_sample_file, "w") as f_samp:
        for val in samples_8bit:
            f_samp.write(f"{val}\n")
            
    # 3. Write Expected Output File (4 bits: N S V F per line for 25 timesteps)
    with open(expected_file, "w") as f_exp:
        f_exp.write(f"# Test Case {case_num:03d}: Test Set Index #{idx}\n")
        f_exp.write(f"# True Class: {class_names[true_label]} ({true_label}) | Predicted Class: {class_names[pred_class]} ({pred_class})\n")
        f_exp.write(f"# Output Spike Counts [N, S, V, F]: {list(spike_counts)}\n")
        f_exp.write("# --- 25 Timesteps of Output Spikes (4 bits: N S V F) ---\n")
        for t in range(TIMESTEPS):
            spk_line = "".join(str(b) for b in out_spikes[t])
            f_exp.write(spk_line + "\n")
            
    match_str = "MATCH" if is_match else "MISS"
    spike_list = [int(s) for s in spike_counts]
    line_out = f"Case {case_num:02d} | True: {class_names[true_label]} | Pred: {class_names[pred_class]} | Spikes [N,S,V,F]: {str(spike_list):<16} | {match_str:<5} | -> {tag}"
    print(line_out)
    manifest_lines.append(line_out + "\n")

# Write master manifest
manifest_path = "test_vectors/test_manifest.txt"
with open(manifest_path, "w", encoding="utf-8") as f_man:
    f_man.writelines(manifest_lines)

total_correct = sum(correct_per_class.values())
overall_acc = (total_correct / total_cases) * 100.0

print("\n" + "=" * 80)
print(f"STEP 3 SUMMARY: {total_cases} TEST VECTORS READY FOR HARDWARE SIMULATION")
print("=" * 80)
print(f"Overall Match Rate: {overall_acc:.2f}% ({total_correct}/{total_cases} correct)")
print("-" * 80)
print(f"{'Class':<25} {'Total Tested':>15} {'Matched':>10} {'Accuracy':>12}")
print("-" * 80)
for c in range(4):
    tot = total_per_class[c]
    cor = correct_per_class[c]
    c_acc = (cor / tot * 100.0) if tot > 0 else 0.0
    print(f"{class_full[c]:<25} {tot:15d} {cor:10d} {c_acc:11.2f}%")

print("=" * 80)
print(f"Master Manifest File: {manifest_path}")
print(f"  - 100 Input spike files  : {DIR_INPUTS}/*_spikes.txt")
print(f"  - 100 Input sample files : {DIR_INPUTS}/*_samples.txt")
print(f"  - 100 Golden answer files: {DIR_EXPECTED}/*_expected.txt")
print("=" * 80)
