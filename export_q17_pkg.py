"""
export_q17_pkg.py
Exports trained float32 weights from the champion checkpoint (lighter_smote_seed999_best.pth)
to strict 8-bit signed Q1.7 integers [-128, +127], and generates the VHDL package:
  - snn_pkg_superspike_q17.vhd (and updates rtl/snn_pkg.vhd)

Architecture:
  - Input: 128 samples
  - Layer 1 (fc1): 128 inputs -> 32 neurons  (4,096 weights)
  - Layer 2 (fc2): 32 inputs  -> 32 neurons  (1,024 weights)
  - Output  (fc3): 32 inputs  -> 4 neurons   (128 weights)
  - Total: 5,248 weights (5.25 KB)
"""

import os
import torch
import numpy as np

CKPT_PATH = "checkpoints/lighter_smote_seed999_best.pth"
if not os.path.exists(CKPT_PATH):
    # Fallback to seed42 if seed999 isn't found
    CKPT_PATH = "checkpoints/lighter_smote_seed42_best.pth"

OUT_VHDL_1 = "snn_pkg_superspike_q17.vhd"
OUT_VHDL_2 = "rtl/snn_pkg.vhd"

print(f"Loading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)

w1 = state_dict["fc1.weight"].cpu().numpy()  # shape: (32, 128)
w2 = state_dict["fc2.weight"].cpu().numpy()  # shape: (32, 32)
w3 = state_dict["fc3.weight"].cpu().numpy()  # shape: (4, 32)

print(f"Loaded Float32 Weights:")
print(f"  fc1.weight shape: {w1.shape} (min: {w1.min():.4f}, max: {w1.max():.4f})")
print(f"  fc2.weight shape: {w2.shape} (min: {w2.min():.4f}, max: {w2.max():.4f})")
print(f"  fc3.weight shape: {w3.shape} (min: {w3.min():.4f}, max: {w3.max():.4f})")

def quantize_to_q17(w_float):
    """Quantizes float32 array into strict 8-bit signed integer Q1.7 [-128, 127]."""
    # Scale factor = 128 (2^7)
    scaled = np.round(w_float * 128.0)
    clamped = np.clip(scaled, -128, 127).astype(np.int8)
    return clamped

w1_q17 = quantize_to_q17(w1)
w2_q17 = quantize_to_q17(w2)
w3_q17 = quantize_to_q17(w3)

print(f"\nQuantized 8-Bit Q1.7 Weights:")
print(f"  w1_q17 shape: {w1_q17.shape} (min: {w1_q17.min()}, max: {w1_q17.max()})")
print(f"  w2_q17 shape: {w2_q17.shape} (min: {w2_q17.min()}, max: {w2_q17.max()})")
print(f"  w3_q17 shape: {w3_q17.shape} (min: {w3_q17.min()}, max: {w3_q17.max()})")

def format_vhdl_array(int8_flat, indent="        ", per_line=16):
    lines = []
    chunk = []
    for i, val in enumerate(int8_flat):
        chunk.append(f"to_signed({int(val):4d}, 8)")
        if len(chunk) == per_line or i == len(int8_flat) - 1:
            lines.append(indent + ", ".join(chunk))
            chunk = []
    return ",\n".join(lines)

w1_flat = w1_q17.flatten()
w2_flat = w2_q17.flatten()
w3_flat = w3_q17.flatten()

vhdl_code = f"""----------------------------------------------------------------------------------
-- Package: SNN Types, Constants, and Trained Q1.7 Weights
-- Project: ECG-SNN-FPGA
-- Architecture: 128 Inputs -> 32 Hidden 1 -> 32 Hidden 2 -> 4 Outputs
-- Timesteps: 25 | Decay: Beta = 0.9375 (Leak Shift = 4)
-- Weights: Strict Q1.7 Signed 8-bit Integers [-128, +127]
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package snn_pkg is

    ---------------------------------------------------------------------------
    -- Network Architecture Constants
    ---------------------------------------------------------------------------
    constant NUM_INPUTS    : integer := 128;   -- ECG window samples
    constant NUM_HIDDEN_1  : integer := 32;    -- Hidden layer 1 neurons
    constant NUM_HIDDEN_2  : integer := 32;    -- Hidden layer 2 neurons
    constant NUM_OUTPUTS   : integer := 4;     -- Output classes (N, S, V, F)
    constant NUM_TIMESTEPS : integer := 25;    -- Simulation timesteps

    ---------------------------------------------------------------------------
    -- LIF Neuron Parameters (Multiplier-Free)
    ---------------------------------------------------------------------------
    constant V_THRESHOLD : signed(15 downto 0) := x"0080";  -- 0.5 in Q8.8 format
    constant LEAK_SHIFT  : integer := 4;  -- beta = 1 - 1/2^4 = 0.9375

    ---------------------------------------------------------------------------
    -- Array Type Definitions
    ---------------------------------------------------------------------------
    type input_array_t is array (0 to NUM_INPUTS - 1) of unsigned(7 downto 0);

    -- Membrane potential registers
    type vmem_array_32_t is array (0 to NUM_HIDDEN_1 - 1) of signed(15 downto 0);
    type vmem_array_4_t  is array (0 to NUM_OUTPUTS - 1)  of signed(15 downto 0);

    type spike_count_4_t is array (0 to NUM_OUTPUTS - 1) of unsigned(7 downto 0);

    -- Weight arrays (Flattened row-major: index = neuron * num_inputs + input)
    type weight_array_h1_t  is array (0 to NUM_INPUTS   * NUM_HIDDEN_1 - 1) of signed(7 downto 0);
    type weight_array_h2_t  is array (0 to NUM_HIDDEN_1  * NUM_HIDDEN_2 - 1) of signed(7 downto 0);
    type weight_array_out_t is array (0 to NUM_HIDDEN_2  * NUM_OUTPUTS  - 1) of signed(7 downto 0);

    ---------------------------------------------------------------------------
    -- Trained Q1.7 Quantized Weights (Signed 8-bit integers)
    ---------------------------------------------------------------------------
    -- Layer 1: 128 inputs x 32 neurons = 4096 weights
    constant WEIGHTS_H1 : weight_array_h1_t := (
{format_vhdl_array(w1_flat)}
    );

    -- Layer 2: 32 inputs x 32 neurons = 1024 weights
    constant WEIGHTS_H2 : weight_array_h2_t := (
{format_vhdl_array(w2_flat)}
    );

    -- Layer 3: 32 inputs x 4 neurons = 128 weights
    constant WEIGHTS_OUT : weight_array_out_t := (
{format_vhdl_array(w3_flat)}
    );

end package snn_pkg;
"""

for target_file in [OUT_VHDL_1, OUT_VHDL_2]:
    with open(target_file, "w") as f:
        f.write(vhdl_code)
    print(f"[OK] Generated: {target_file}")

print("\nDone! Weights successfully exported into VHDL package.")
