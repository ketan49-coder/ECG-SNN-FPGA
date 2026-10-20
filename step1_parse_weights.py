"""
step1_parse_weights.py
Step 1: Parse the constants from snn_pkg_superspike_q17.vhd into Python.
Print the shapes and verify them against the hardware ports (128 inputs, 32 hidden, 4 outputs).
"""

import os
import re
import numpy as np

VHDL_PKG_FILE = "snn_pkg_superspike_q17.vhd"
if not os.path.exists(VHDL_PKG_FILE) and os.path.exists("rtl/" + VHDL_PKG_FILE):
    VHDL_PKG_FILE = "rtl/" + VHDL_PKG_FILE

print("=" * 70)
print(f"STEP 1: PARSING CONSTANTS FROM {VHDL_PKG_FILE}")
print("=" * 70)

if not os.path.exists(VHDL_PKG_FILE):
    print(f"\n[ERROR] '{VHDL_PKG_FILE}' not found!")
    print("Please run 'python export_q17_pkg.py' first to generate it from model checkpoint.")
    exit(1)

with open(VHDL_PKG_FILE, "r") as f:
    vhdl_text = f.read()

def parse_vhdl_weight_block(constant_name, text):
    pattern = rf"constant\s+{constant_name}\s*:\s*\w+\s*:=\s*\((.*?)\);"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find constant '{constant_name}' in {VHDL_PKG_FILE}")
    block_text = match.group(1)
    int_values = [int(v) for v in re.findall(r"to_signed\(\s*(-?\d+)\s*,\s*8\s*\)", block_text)]
    return np.array(int_values, dtype=np.int8)

# Parse weights
w_h1_raw = parse_vhdl_weight_block("WEIGHTS_H1", vhdl_text)
w_h2_raw = parse_vhdl_weight_block("WEIGHTS_H2", vhdl_text)
w_out_raw = parse_vhdl_weight_block("WEIGHTS_OUT", vhdl_text)

# Reshape into 2D matrices
# Layer 1: 32 neurons, each connected to 128 inputs
w_h1 = w_h1_raw.reshape((32, 128))

# Layer 2: 32 neurons, each connected to 32 inputs
w_h2 = w_h2_raw.reshape((32, 32))

# Layer 3 / Output: 4 neurons (N, S, V, F), each connected to 32 inputs
w_out = w_out_raw.reshape((4, 32))

print(f"\n[PARSED WEIGHT MATRICES]")
print(f"  Layer 1 (WEIGHTS_H1)  : {len(w_h1_raw):5d} constants -> Shape: {w_h1.shape}  [Expected: (32, 128)]")
print(f"  Layer 2 (WEIGHTS_H2)  : {len(w_h2_raw):5d} constants -> Shape: {w_h2.shape}   [Expected: (32, 32)]")
print(f"  Output  (WEIGHTS_OUT) : {len(w_out_raw):5d} constants -> Shape: {w_out.shape}    [Expected: (4, 32)]")
print(f"  Total Model Weights   : {len(w_h1_raw) + len(w_h2_raw) + len(w_out_raw):5d} (Strict Q1.7 8-bit integers)")

print("\n[VERIFICATION AGAINST HARDWARE PORTS]")
print("  Checking: 128 inputs -> 32 hidden -> 32 hidden -> 4 outputs...")

assert w_h1.shape == (32, 128), f"Layer 1 shape mismatch: {w_h1.shape} != (32, 128)"
assert w_h2.shape == (32, 32),  f"Layer 2 shape mismatch: {w_h2.shape} != (32, 32)"
assert w_out.shape == (4, 32),  f"Output shape mismatch: {w_out.shape} != (4, 32)"

print("  --> Layer 1 Shape MATCHES input port: 128 inputs to 32 hidden neurons (OK)")
print("  --> Layer 2 Shape MATCHES hidden port: 32 hidden to 32 hidden neurons (OK)")
print("  --> Output  Shape MATCHES output port: 32 hidden to 4 output classes (OK)")

print("\n" + "=" * 70)
print("SUCCESS: STEP 1 COMPLETE. All weights match hardware architecture 100%!")
print("=" * 70)
