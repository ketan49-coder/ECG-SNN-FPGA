import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

# Import network definition
from train_snn_superspike import ECG_SuperSpike

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
# Checks Google Drive first (Colab), otherwise current directory
if os.path.exists("/content/drive/MyDrive/SNN_Models/snn_superspike_best.pth"):
    MODEL_PATH = "/content/drive/MyDrive/SNN_Models/snn_superspike_best.pth"
elif os.path.exists("snn_superspike_best.pth"):
    MODEL_PATH = "snn_superspike_best.pth"
elif os.path.exists("python/snn_superspike_best.pth"):
    MODEL_PATH = "python/snn_superspike_best.pth"
else:
    MODEL_PATH = "snn_superspike_best.pth"

OUTPUT_VHDL_PKG = "snn_pkg_superspike_q17.vhd"
CLASS_NAMES = ['Normal (N)', 'Supra (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data():
    X_test_path = os.path.join(DATA_DIR, "X_test.npy")
    y_test_path = os.path.join(DATA_DIR, "y_test.npy")
    if not os.path.exists(X_test_path):
        # Fallback to local files if not in data/
        X_test_path = "X_test.npy"
        y_test_path = "y_test.npy"
        
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    return DataLoader(ds, batch_size=128, shuffle=False)

def run_evaluation(net, loader):
    net.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            spk_rec = net(data)
            preds = spk_rec.sum(0).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = 100.0 * np.mean(all_preds == all_targets)
    return acc, all_targets, all_preds

def quantize_to_q17(tensor):
    """
    Quantizes a floating point tensor into strict Q1.7 8-bit signed format.
    Q1.7: 1 sign bit, 7 fractional bits.
    Integer range: -128 to +127
    Representable float range: -1.0 to +127/128 (~0.9922)
    Scale factor: 128.0
    """
    int_vals = torch.round(tensor * 128.0)
    int_clamped = torch.clamp(int_vals, -128.0, 127.0)
    float_quant = int_clamped / 128.0
    return float_quant, int_clamped.to(torch.int8)

def format_vhdl_array(int8_numpy_flat, indent="        ", elements_per_line=16):
    """Formats a 1D int8 array into valid VHDL signed literal array syntax."""
    lines = []
    current_line = []
    for i, val in enumerate(int8_numpy_flat):
        current_line.append(f'to_signed({int(val):4d}, 8)')
        if len(current_line) == elements_per_line or i == len(int8_numpy_flat) - 1:
            lines.append(indent + ", ".join(current_line))
            current_line = []
    return ",\n".join(lines)

def main():
    print(f"\n=======================================================")
    print(f"STEP 1: Float32 vs Q1.7 Quantization Benchmark")
    print(f"=======================================================")
    print(f"Loading weights from: {MODEL_PATH}")
    
    test_loader = load_test_data()
    net = ECG_SuperSpike().to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Float32 Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    acc_flt, targets_flt, preds_flt = run_evaluation(net, test_loader)
    print(f"\n[1] Float32 Baseline Accuracy: {acc_flt:.2f}%")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. Q1.7 Quantization
    # ─────────────────────────────────────────────────────────────────────────
    q_weights_int = {}
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'weight' in name:
                flt_q, int_q = quantize_to_q17(param.data)
                param.data.copy_(flt_q)
                q_weights_int[name] = int_q.cpu().numpy()
                
    # ─────────────────────────────────────────────────────────────────────────
    # 3. Quantized Model Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    acc_q, targets_q, preds_q = run_evaluation(net, test_loader)
    drop = acc_flt - acc_q
    
    print(f"[2] 8-bit Q1.7 Model Accuracy: {acc_q:.2f}%")
    print(f"[3] Accuracy Drop from Q1.7  : {drop:+.2f}%\n")
    
    print("=" * 65)
    print("CLASSIFICATION REPORT UNDER 8-BIT Q1.7 QUANTIZATION:")
    print("=" * 65)
    print(classification_report(targets_q, preds_q, target_names=CLASS_NAMES, digits=4))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. Generate Hardware VHDL Package with Embedded Weights
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nGenerating VHDL Hardware Package: {OUTPUT_VHDL_PKG}...")
    w1_flat = q_weights_int['fc1.weight'].flatten()  # 32 * 128 = 4096
    w2_flat = q_weights_int['fc2.weight'].flatten()  # 32 * 32 = 1024
    w3_flat = q_weights_int['fc3.weight'].flatten()  # 5 * 32 = 160
    
    vhdl_content = f"""----------------------------------------------------------------------------------
-- Package: SNN Types, Constants, and Trained Q1.7 Weights
-- Project: ECG-SNN-FPGA (Spartan-7 Multiplier-Free SNN)
-- Topology: 128 Inputs -> 32 Hidden 1 -> 32 Hidden 2 -> 5 Outputs
-- Timesteps: 25 | Decay: Beta = 0.9375 (Leak Shift = 4)
-- Model Accuracy: Float32 = {acc_flt:.2f}%, Q1.7 Quantized = {acc_q:.2f}%
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
    constant NUM_OUTPUTS   : integer := 5;     -- Output classes (N, S, V, F, Q)
    constant NUM_TIMESTEPS : integer := 25;    -- Simulation timesteps

    ---------------------------------------------------------------------------
    -- LIF Neuron Parameters (Multiplier-Free)
    ---------------------------------------------------------------------------
    constant V_THRESHOLD : signed(15 downto 0) := x"0080";  -- 0.5 in Q8.8
    constant LEAK_SHIFT  : integer := 4;  -- beta = 1 - 1/2^4 = 0.9375

    ---------------------------------------------------------------------------
    -- Array Type Definitions
    ---------------------------------------------------------------------------
    type input_array_t is array (0 to NUM_INPUTS - 1) of unsigned(7 downto 0);

    -- Membrane potential registers for 32 hidden neurons and 5 output neurons
    type vmem_array_32_t is array (0 to 31) of signed(15 downto 0);
    type vmem_array_5_t  is array (0 to 4)  of signed(15 downto 0);

    type spike_count_5_t is array (0 to 4) of unsigned(7 downto 0);

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

    -- Layer 3: 32 inputs x 5 neurons = 160 weights
    constant WEIGHTS_OUT : weight_array_out_t := (
{format_vhdl_array(w3_flat)}
    );

end package snn_pkg;
"""
    with open(OUTPUT_VHDL_PKG, "w") as f:
        f.write(vhdl_content)
        
    print(f"Successfully generated {OUTPUT_VHDL_PKG} with all 5,280 Q1.7 weights!")
    print("Ready to copy directly into Vivado for Monday's deployment.")

if __name__ == "__main__":
    main()
