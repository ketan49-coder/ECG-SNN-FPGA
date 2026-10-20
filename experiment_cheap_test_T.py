"""
experiment_cheap_test_T.py
Tests whether increasing timesteps T (T=25, 35, 50) reduces Bernoulli sampling variance
and recovers accuracy in the exact LFSR hardware twin without retraining.
"""

import os
import re
import numpy as np

# Parse weights
VHDL_PKG = "rtl/snn_pkg.vhd"
with open(VHDL_PKG, "r") as f:
    text = f.read()

def parse_vhdl_weights(name):
    m = re.search(rf"constant\s+{name}\s*:\s*\w+\s*:=\s*\((.*?)\);", text, re.DOTALL)
    vals = [int(v) for v in re.findall(r"to_signed\(\s*(-?\d+)\s*,\s*8\s*\)", m.group(1))]
    return np.array(vals, dtype=np.int32)

W1 = parse_vhdl_weights("WEIGHTS_H1").reshape((32, 128))
W2 = parse_vhdl_weights("WEIGHTS_H2").reshape((32, 32))
W3 = parse_vhdl_weights("WEIGHTS_OUT").reshape((4, 32))

THRESHOLD = 64
LEAK_SHIFT = 4

class HardwareLFSR8:
    def __init__(self, seed=0xA5):
        self.reg = seed & 0xFF
    def reset(self, seed=0xA5):
        self.reg = seed & 0xFF
    def step(self):
        val = self.reg
        b7 = (self.reg >> 7) & 1
        b5 = (self.reg >> 5) & 1
        b4 = (self.reg >> 4) & 1
        b3 = (self.reg >> 3) & 1
        fb = b7 ^ b5 ^ b4 ^ b3
        self.reg = ((self.reg << 1) & 0xFF) | fb
        return val

def argmax_4_hardware(counts):
    max_val = counts[0]
    max_idx = 0
    if counts[1] > max_val: max_val = counts[1]; max_idx = 1
    if counts[2] > max_val: max_val = counts[2]; max_idx = 2
    if counts[3] > max_val: max_val = counts[3]; max_idx = 3
    return max_idx

def run_twin(samples_8bit, T, lfsr):
    v1 = np.zeros(32, dtype=np.int32)
    v2 = np.zeros(32, dtype=np.int32)
    v3 = np.zeros(4,  dtype=np.int32)
    counts = np.zeros(4, dtype=np.int32)

    for t in range(T):
        spikes_in = np.zeros(128, dtype=np.int32)
        for i in range(128):
            lfsr_val = lfsr.step()
            if samples_8bit[i] > lfsr_val:
                spikes_in[i] = 1

        accum1 = np.dot(W1, spikes_in)
        v1_leaked = v1 - np.right_shift(v1, LEAK_SHIFT)
        v1_new = v1_leaked + accum1
        s1 = (v1_new >= THRESHOLD).astype(np.int32)
        v1 = np.where(s1 == 1, 0, v1_new)

        accum2 = np.dot(W2, s1)
        v2_leaked = v2 - np.right_shift(v2, LEAK_SHIFT)
        v2_new = v2_leaked + accum2
        s2 = (v2_new >= THRESHOLD).astype(np.int32)
        v2 = np.where(s2 == 1, 0, v2_new)

        accum3 = np.dot(W3, s2)
        v3_leaked = v3 - np.right_shift(v3, LEAK_SHIFT)
        v3_new = v3_leaked + accum3
        s3 = (v3_new >= THRESHOLD).astype(np.int32)
        v3 = np.where(s3 == 1, 0, v3_new)

        counts += s3

    return argmax_4_hardware(counts)

if __name__ == "__main__":
    data = np.load("sparrow_test_set.npz")
    X_test, y_test = data['X_test'], data['y_test']
    N_EVAL = 1000

    print("=" * 65)
    print(f"CHEAP TEST: TIMESTEP SWEEP ON EXACT LFSR TWIN (N={N_EVAL})")
    print("=" * 65)

    for T in [125, 150]:
        lfsr = HardwareLFSR8(seed=0xA5)
        cmat = np.zeros((4, 4), dtype=int)
        for i in range(N_EVAL):
            s8 = np.clip(np.round(X_test[i] * 255.0), 0, 255).astype(np.uint8)
            pred = run_twin(s8, T, lfsr)
            cmat[y_test[i], pred] += 1
        
        acc = np.trace(cmat) / N_EVAL * 100.0
        n_acc = cmat[0,0] / np.sum(cmat[0,:]) * 100
        s_acc = cmat[1,1] / np.sum(cmat[1,:]) * 100 if np.sum(cmat[1,:])>0 else 0
        v_acc = cmat[2,2] / np.sum(cmat[2,:]) * 100 if np.sum(cmat[2,:])>0 else 0
        f_acc = cmat[3,3] / np.sum(cmat[3,:]) * 100 if np.sum(cmat[3,:])>0 else 0

        print(f"T = {T:2d} Timesteps | Overall Acc: {acc:5.2f}% | N: {n_acc:5.1f}% | S: {s_acc:4.1f}% | V: {v_acc:5.1f}% | F: {f_acc:4.1f}%")
        print(f"      Confusion Matrix [N, S, V, F]:")
        for row in cmat:
            print(f"        {row}")
        print("-" * 65)
