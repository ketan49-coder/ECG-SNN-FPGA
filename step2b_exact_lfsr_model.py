"""
step2b_exact_lfsr_model.py
Bit-true cycle-accurate digital twin of the RTL hardware (snn_top.vhd + lfsr_8bit.vhd).

Hardware specifications modeled:
  - 8-bit Fibonacci LFSR (seed 0xA5, taps at 7, 5, 4, 3)
  - Single shared LFSR across all 128 inputs, advancing each sample (128x per timestep)
  - Strict sample > LFSR comparison
  - Layer 1: Multiplier-free accumulation of Q1.7 weights on input spikes
  - Layer 2 & 3: Multiplier-free LIF (leak V - (V >> 4), threshold 64, zero reset)
  - Hardware argmax_4 with strict '>' priority tie-breaker (Class 0 wins ties)
"""

import os
import re
import numpy as np

# ── 1. Parse Q1.7 weights from the VHDL package ──────────────────────────────
VHDL_PKG = "rtl/snn_pkg.vhd"
if not os.path.exists(VHDL_PKG):
    VHDL_PKG = "snn_pkg_superspike_q17.vhd"

with open(VHDL_PKG, "r") as f:
    text = f.read()

def parse_vhdl_weights(name):
    m = re.search(rf"constant\s+{name}\s*:\s*\w+\s*:=\s*\((.*?)\);", text, re.DOTALL)
    vals = [int(v) for v in re.findall(r"to_signed\(\s*(-?\d+)\s*,\s*8\s*\)", m.group(1))]
    return np.array(vals, dtype=np.int32)

W1 = parse_vhdl_weights("WEIGHTS_H1").reshape((32, 128))  # (32, 128)
W2 = parse_vhdl_weights("WEIGHTS_H2").reshape((32, 32))   # (32, 32)
W3 = parse_vhdl_weights("WEIGHTS_OUT").reshape((4, 32))   # (4, 32)

THRESHOLD = 64
LEAK_SHIFT = 4
TIMESTEPS = 25

class HardwareLFSR8:
    """
    Exact replica of rtl/lfsr_8bit.vhd:
      process(clk)
        variable feedback : std_logic;
      begin
        if rising_edge(clk) then
          if rst = '1' then
            lfsr_reg <= x"A5";
          elsif en = '1' then
            feedback := lfsr_reg(7) xor lfsr_reg(5) xor lfsr_reg(4) xor lfsr_reg(3);
            lfsr_reg <= lfsr_reg(6 downto 0) & feedback;
          end if;
        end if;
      end process;
      lfsr_out <= lfsr_reg;
    """
    def __init__(self, seed=0xA5):
        self.reg = seed & 0xFF

    def reset(self, seed=0xA5):
        self.reg = seed & 0xFF

    def step(self):
        """Advances LFSR by 1 clock cycle when en='1' and returns current output."""
        val = self.reg
        b7 = (self.reg >> 7) & 1
        b5 = (self.reg >> 5) & 1
        b4 = (self.reg >> 4) & 1
        b3 = (self.reg >> 3) & 1
        feedback = b7 ^ b5 ^ b4 ^ b3
        self.reg = ((self.reg << 1) & 0xFF) | feedback
        return val

def argmax_4_hardware(counts):
    """
    Exact replica of rtl/argmax_4.vhd:
      max_val := val_0; max_idx := "00";
      if val_1 > max_val then max_val := val_1; max_idx := "01"; end if;
      if val_2 > max_val then max_val := val_2; max_idx := "10"; end if;
      if val_3 > max_val then max_val := val_3; max_idx := "11"; end if;
    """
    max_val = counts[0]
    max_idx = 0
    if counts[1] > max_val:
        max_val = counts[1]
        max_idx = 1
    if counts[2] > max_val:
        max_val = counts[2]
        max_idx = 2
    if counts[3] > max_val:
        max_val = counts[3]
        max_idx = 3
    return max_idx

def run_hardware_snn_inference(samples_8bit, lfsr=None, reset_lfsr=True):
    """
    Simulates the exact VHDL RTL snn_top hardware inference for one heartbeat.
    samples_8bit : array of 128 uint8 values (0..255)
    """
    if lfsr is None:
        lfsr = HardwareLFSR8(seed=0xA5)
    elif reset_lfsr:
        lfsr.reset(seed=0xA5)

    v1 = np.zeros(32, dtype=np.int32)
    v2 = np.zeros(32, dtype=np.int32)
    v3 = np.zeros(4,  dtype=np.int32)
    
    spike_count_out = np.zeros(4, dtype=np.int32)
    out_spikes_per_ts = []

    for t in range(TIMESTEPS):
        # 1. S_ENCODE: 128 cycles, LFSR advances each cycle
        spikes_encoded = np.zeros(128, dtype=np.int32)
        for i in range(128):
            lfsr_val = lfsr.step()
            if samples_8bit[i] > lfsr_val:
                spikes_encoded[i] = 1
            else:
                spikes_encoded[i] = 0

        # 2. S_PROC_H1: 32 neurons, sequential accumulation of spikes
        # Multiplier-free: accum = sum(W1 * spikes_encoded)
        accum_h1 = np.dot(W1, spikes_encoded)
        v1_leaked = v1 - np.right_shift(v1, LEAK_SHIFT)
        v1_new = v1_leaked + accum_h1
        s1 = (v1_new >= THRESHOLD).astype(np.int32)
        v1 = np.where(s1 == 1, 0, v1_new)

        # 3. S_PROC_H2: 32 neurons, sequential accumulation of s1 spikes
        accum_h2 = np.dot(W2, s1)
        v2_leaked = v2 - np.right_shift(v2, LEAK_SHIFT)
        v2_new = v2_leaked + accum_h2
        s2 = (v2_new >= THRESHOLD).astype(np.int32)
        v2 = np.where(s2 == 1, 0, v2_new)

        # 4. S_PROC_OUT: 4 neurons, sequential accumulation of s2 spikes
        accum_out = np.dot(W3, s2)
        v3_leaked = v3 - np.right_shift(v3, LEAK_SHIFT)
        v3_new = v3_leaked + accum_out
        s3 = (v3_new >= THRESHOLD).astype(np.int32)
        v3 = np.where(s3 == 1, 0, v3_new)

        # Track spike counts
        spike_count_out += s3
        out_spikes_per_ts.append(s3)

    pred_class = argmax_4_hardware(spike_count_out)
    return pred_class, spike_count_out, np.array(out_spikes_per_ts)

if __name__ == "__main__":
    DATA_PATH = "sparrow_test_set.npz"
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Test set '{DATA_PATH}' not found!")
        exit(1)

    data = np.load(DATA_PATH)
    X_test, y_test = data['X_test'], data['y_test']
    print(f"Loaded Test Dataset: {len(X_test)} heartbeats")

    # Evaluate on first 2,000 beats to get fast, rigorous statistical accuracy
    N_EVAL = min(2000, len(X_test))
    print(f"\nEvaluating EXACT Hardware LFSR Model on N={N_EVAL} test heartbeats...")

    classes_short = ['N', 'S', 'V', 'F']
    classes_full  = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']
    confusion_mat = np.zeros((4, 4), dtype=int)
    
    # We will test both modes:
    # Mode A: Free-running LFSR (matching continuous RTL running)
    # Mode B: Per-beat reset LFSR
    lfsr_continuous = HardwareLFSR8(seed=0xA5)

    for i in range(N_EVAL):
        beat_float = X_test[i]
        samples_8bit = np.clip(np.round(beat_float * 255.0), 0, 255).astype(np.uint8)
        true_label = y_test[i]

        pred_label, spk_counts, _ = run_hardware_snn_inference(samples_8bit, lfsr=lfsr_continuous, reset_lfsr=False)
        confusion_mat[true_label, pred_label] += 1

        if (i + 1) % 500 == 0 or (i + 1) == N_EVAL:
            print(f"  Processed {i+1:5d} / {N_EVAL} beats...")

    total_correct = int(np.trace(confusion_mat))
    acc = (total_correct / N_EVAL) * 100.0

    print("\n" + "=" * 70)
    print(f"EXACT RTL LFSR HARDWARE BENCHMARK (N={N_EVAL} SAMPLES)")
    print("=" * 70)
    print(f"Hardware-Level Overall Accuracy: {acc:.2f}% ({total_correct}/{N_EVAL} correct)")
    print("-" * 70)
    print(f"{'Class':<25} {'Total Beats':>12} {'Correct':>10} {'Accuracy':>12}")
    print("-" * 70)

    for c in range(4):
        n_class_total = int(np.sum(confusion_mat[c, :]))
        n_class_correct = int(confusion_mat[c, c])
        c_acc = (n_class_correct / n_class_total * 100.0) if n_class_total > 0 else 0.0
        print(f"{classes_full[c]:<25} {n_class_total:12d} {n_class_correct:10d} {c_acc:11.2f}%")

    print("=" * 70)
    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    print(f"         Pred N   Pred S   Pred V   Pred F")
    for c in range(4):
        row_str = "  ".join(f"{confusion_mat[c, p]:6d}" for p in range(4))
        print(f"True {classes_short[c]}:  {row_str}")

    # Now verify Test 1 and Test 2 from Vivado testbench
    print("\n" + "=" * 70)
    print("TESTBENCH BEATS DIRECT VALIDATION:")
    print("=" * 70)
    
    # Patient 1: Normal beat from snn_top_tb.vhd
    beat_N = np.array([
        71, 70, 69, 69, 69, 74, 75, 75, 74, 75,
        77, 79, 80, 80, 81, 82, 83, 86, 86, 85,
        84, 84, 84, 83, 83, 80, 78, 75, 74, 75,
        75, 74, 72, 71, 72, 72, 73, 71, 70, 69,
        70, 71, 71, 71, 70, 70, 70, 72, 72, 71,
        70, 71, 71, 71, 70, 70, 73, 79, 91, 107,
        128, 154, 179, 202, 219, 221, 201, 170, 134, 102,
        80, 73, 70, 70, 69, 68, 69, 73, 74, 72,
        70, 69, 69, 71, 71, 70, 69, 69, 71, 71,
        73, 72, 71, 70, 71, 72, 72, 72, 72, 72,
        72, 74, 74, 72, 73, 73, 75, 76, 78, 78,
        77, 76, 79, 81, 81, 80, 79, 79, 82, 84,
        85, 84, 84, 85, 86, 89, 90, 91
    ], dtype=np.uint8)

    # Patient 2: Ventricular beat from snn_top_tb.vhd
    beat_V = np.array([
        84, 83, 82, 80, 78, 77, 77, 77, 78, 78,
        77, 76, 75, 74, 73, 74, 74, 75, 76, 76,
        75, 75, 75, 76, 76, 76, 76, 76, 76, 77,
        78, 79, 81, 82, 83, 83, 84, 84, 84, 85,
        84, 84, 83, 81, 79, 78, 78, 79, 82, 88,
        95, 103, 112, 121, 131, 140, 149, 156, 163, 168,
        172, 175, 178, 181, 183, 185, 185, 184, 182, 180,
        178, 175, 174, 172, 170, 168, 164, 159, 153, 145,
        136, 127, 116, 106, 97, 88, 79, 72, 66, 60,
        56, 53, 51, 48, 46, 45, 43, 43, 43, 43,
        42, 42, 40, 39, 38, 38, 38, 37, 36, 35,
        34, 33, 32, 31, 31, 31, 31, 30, 30, 29,
        28, 28, 28, 27, 27, 27, 27, 28
    ], dtype=np.uint8)

    tb_lfsr = HardwareLFSR8(seed=0xA5)
    pred_N, counts_N, spk_seq_N = run_hardware_snn_inference(beat_N, lfsr=tb_lfsr, reset_lfsr=False)
    pred_V, counts_V, spk_seq_V = run_hardware_snn_inference(beat_V, lfsr=tb_lfsr, reset_lfsr=False)

    print(f"Patient 1 (Normal)     -> Pred Class: {classes_short[pred_N]} ({pred_N}) | Spike Counts [N,S,V,F]: {list(counts_N)}")
    print(f"Patient 2 (Ventricular) -> Pred Class: {classes_short[pred_V]} ({pred_V}) | Spike Counts [N,S,V,F]: {list(counts_V)}")
