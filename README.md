# CardioSpike-FPGA: SNN-Based ECG Classification

A hardware-efficient Spiking Neural Network (SNN) implemented in VHDL for real-time ECG arrhythmia classification, targeting FPGAs. 

This project bridges software-based neuromorphic training with digital hardware deployment, classifying heartbeats from the MIT-BIH Arrhythmia Database into 5 standard AAMI classes.

## Project Overview

Traditional Artificial Neural Networks (ANNs) rely on resource-heavy digital multipliers for synaptic operations. This project implements a **multiplier-free** Leaky Integrate-and-Fire (LIF) SNN. By using rate-coded binary spikes and a shift-based leak mechanism, all neural computations are reduced to simple addition and bit-shift operations, making it highly efficient for edge deployment on FPGAs.

### Key Features
* **Architecture:** 128 (Input) → 16 (Hidden-1) → 16 (Hidden-2) → 5 (Output) LIF Neurons
* **Dataset:** MIT-BIH Arrhythmia Database (5 AAMI classes: N, S, V, F, Q)
* **Encoding:** LFSR-based rate coding (converting ECG amplitudes to spike trains)
* **Hardware Efficiency:** 0 DSP slices used (multiplier-free)
* **Precision:** 8-bit Q1.7 fixed-point weights (stored in BRAM), 16-bit Q8.8 membrane potentials
* **Training:** Google Colab (GPU) with snnTorch surrogate gradients
* **Languages:** Python (snnTorch) for offline training, VHDL for hardware inference

## Repository Structure

```
├── python/
│   ├── preprocess.py        # ECG signal extraction, R-peak windowing, AAMI labeling
│   ├── augment_data.py      # Minority-class augmentation (noise, shift, scale, wander)
│   ├── train_snn.py         # SNN training (128→16→16→5, snnTorch + surrogate gradients)
│   ├── quantize_weights.py  # Post-training 8-bit Q1.7 quantization → .coe / .hex export
│   ├── explore_data.py      # Dataset visualization and class distribution plots
│   └── requirements.txt     # Python dependencies
├── src/                     # VHDL RTL source code (LIF neuron, Controller, FSM)
├── tb/                      # VHDL testbenches and test vectors
├── constraints/             # Xilinx XDC constraint files
├── sim/                     # Vivado simulation TCL scripts
└── docs/                    # Project reports and block diagrams
```

## Workflow

1. **Preprocessing (Python):** ECG signals from MIT-BIH are segmented into 128-sample windows centered on R-peaks, normalized to [0, 1], and labeled into 5 AAMI classes. Minority classes are augmented to combat class imbalance.
2. **Offline Training (Colab/Python):** The preprocessed data is used to train a 3-layer LIF SNN (128→16→16→5) using snnTorch with surrogate gradient descent. Inverse-frequency class weighting handles the ~71% Normal-class dominance.
3. **Quantization (Python):** Trained float32 weights are quantized to 8-bit Q1.7 fixed-point (range [-1, +0.9921875], resolution 0.0078125). Quantized weights are exported as Xilinx `.coe` files (for BRAM initialization) and `.hex` files (for simulation/testbenches). A quantization error report is generated to verify accuracy is preserved.
4. **Hardware Inference (VHDL):** The quantized weights are loaded into FPGA Block RAM. The VHDL inference engine rate-encodes the incoming ECG signal, processes it through the time-multiplexed hidden and output layers over 16 timesteps, and outputs the predicted class via an Argmax function.

## Tools Required
* **Training:** Google Colab (free GPU) or local Python 3.8+
* **Dependencies:** PyTorch, snnTorch, WFDB, NumPy, scikit-learn, matplotlib
* **Hardware:** Xilinx Vivado Design Suite
