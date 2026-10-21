import os
import numpy as np

# Resolve project root regardless of where the command is executed from
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, 'rtl')):
    ROOT_DIR = script_dir
elif os.path.exists(os.path.join(script_dir, '..', 'rtl')):
    ROOT_DIR = os.path.abspath(os.path.join(script_dir, '..'))
else:
    ROOT_DIR = os.getcwd()

# Find weights file
weights_path = os.path.join(ROOT_DIR, 'weights.npz')
if not os.path.exists(weights_path):
    # Check in parent or checkpoints
    for alt in [os.path.join(ROOT_DIR, 'best_real_data_t75_seed123_weights.npz'),
                os.path.join(ROOT_DIR, 'checkpoints', 'lighter_smote_seed123_weights.npz')]:
        if os.path.exists(alt):
            weights_path = alt
            break

print(f"Loading weights from: {weights_path}")
w = np.load(weights_path)
W1 = w['W1'] if 'W1' in w else w['w1']
W2 = w['W2'] if 'W2' in w else w['w2']
W3 = w['W3'] if 'W3' in w else w['w3']

w1_flat = W1.flatten()
w2_flat = W2.flatten()
w3_flat = W3.flatten()

def format_1d(arr, per_line=16):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        line = '        ' + ', '.join([f'to_signed({int(x):>5}, 8)' for x in chunk])
        lines.append(line)
    return ',\n'.join(lines)

header = """----------------------------------------------------------------------------------
-- Package: snn_pkg
-- Project: ECG-SNN-FPGA (Spartan-7 xc7s15)
-- Description:
--   Global configuration parameters, constants, and trained Q1.7 weights
--   for the 4-class ECG Spiking Neural Network (128 -> 32 -> 32 -> 4).
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
    constant NUM_TIMESTEPS : integer := 75;    -- Simulation timesteps (T=75)

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
"""

mid1 = """    );

    -- Layer 2: 32 inputs x 32 neurons = 1024 weights
    constant WEIGHTS_H2 : weight_array_h2_t := (
"""

mid2 = """    );

    -- Output Layer: 32 inputs x 4 neurons = 128 weights
    constant WEIGHTS_OUT : weight_array_out_t := (
"""

footer = """    );

end package snn_pkg;
"""

vhdl_pkg = header + format_1d(w1_flat) + mid1 + format_1d(w2_flat) + mid2 + format_1d(w3_flat) + footer

target_paths = [
    os.path.join(ROOT_DIR, 'snn_pkg.vhd'),
    os.path.join(ROOT_DIR, 'rtl', 'snn_pkg.vhd'),
    os.path.join(ROOT_DIR, 'SNN-FPGA-ECG', 'SNN-FPGA-ECG.srcs', 'sources_1', 'imports', 'rtl', 'snn_pkg.vhd')
]

for path in target_paths:
    dir_name = os.path.dirname(path)
    if os.path.exists(dir_name):
        with open(path, 'w') as f:
            f.write(vhdl_pkg)
        print(f"Updated: {path}")

print("\nSUCCESS: All snn_pkg.vhd files updated successfully!")
