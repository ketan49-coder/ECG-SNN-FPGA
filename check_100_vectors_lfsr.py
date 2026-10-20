"""
check_100_vectors_lfsr.py
Runs the exact RTL LFSR hardware model on all 100 test vectors in test_vectors/inputs/.
Dumps exact 4-class spike counts [N, S, V, F], winner, and true label.
"""

import os
import glob
import numpy as np
from step2b_exact_lfsr_model import run_hardware_snn_inference, HardwareLFSR8, argmax_4_hardware

DIR_INPUTS = "test_vectors/inputs"
sample_files = sorted(glob.glob(os.path.join(DIR_INPUTS, "*_samples.txt")))

print("=" * 75)
print(f"RUNNING EXACT RTL LFSR TWIN ON ALL {len(sample_files)} TEST VECTORS")
print("=" * 75)
print(f"{'Case':<10} {'True':<6} {'Pred':<6} {'Spike Counts [N, S, V, F]':<30} {'Match?'}")
print("-" * 75)

lfsr = HardwareLFSR8(seed=0xA5)
class_map = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
class_inv = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}

results = []
correct_count = 0

for sf in sample_files:
    basename = os.path.basename(sf)
    # format: beat_000_N_samples.txt
    parts = basename.split("_")
    case_id = parts[1]
    true_cls_str = parts[2]
    true_cls = class_map[true_cls_str]

    with open(sf, "r") as f:
        samples_8bit = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.uint8)

    # Free-running LFSR matching hardware
    pred_cls, counts, out_spk = run_hardware_snn_inference(samples_8bit, lfsr=lfsr, reset_lfsr=False)
    
    is_match = (pred_cls == true_cls)
    if is_match:
        correct_count += 1
        
    match_str = "[OK]" if is_match else "[MISMATCH]"
    print(f"Beat {case_id:<5} {true_cls_str:<6} {class_inv[pred_cls]:<6} {str(list(counts)):<30} {match_str}")
    results.append((case_id, true_cls_str, class_inv[pred_cls], list(counts), is_match))

acc = correct_count / len(sample_files) * 100.0
print("-" * 75)
print(f"Total Matches with True Clinical Labels: {correct_count}/{len(sample_files)} ({acc:.1f}%)")
print("=" * 75)

# Save golden hardware spike counts
out_catalog = "test_vectors/lfsr_hardware_spike_counts.txt"
with open(out_catalog, "w") as f:
    f.write("# Exact RTL LFSR Hardware Spike Counts for 100 Test Vectors\n")
    f.write(f"# Case, True, Pred, Counts [N S V F], Match\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
print(f"Saved exact hardware spike counts to: {out_catalog}")
