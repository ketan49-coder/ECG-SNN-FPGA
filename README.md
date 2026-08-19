# CardioSpike-FPGA: SNN-Based ECG Classification

A hardware-efficient Spiking Neural Network (SNN) implemented in VHDL for real-time ECG arrhythmia classification, targeting FPGAs. 

This project bridges software-based neuromorphic training with digital hardware deployment, classifying heartbeats from the MIT-BIH Arrhythmia Database into 5 standard AAMI classes.

## Project Overview

Traditional Artificial Neural Networks (ANNs) rely on resource-heavy digital multipliers for synaptic operations. This project implements a **multiplier-free** Leaky Integrate-and-Fire (LIF) SNN. By using rate-coded binary spikes and a shift-based leak mechanism, all neural computations are reduced to simple addition and bit-shift operations, making it highly efficient for edge deployment on FPGAs.

### Key Features
* **Architecture:** 128 (Input) → 32 (Hidden) → 5 (Output) LIF Neurons
* **Dataset:** MIT-BIH Arrhythmia Database (5 AAMI classes: N, S, V, F, Q)
* **Encoding:** LFSR-based rate coding (converting ECG amplitudes to spike trains)
* **Hardware Efficiency:** 0 DSP slices used (multiplier-free)
* **Precision:** 8-bit Q1.7 fixed-point weights (stored in BRAM), 16-bit Q8.8 membrane potentials
* **Languages:** Python (snnTorch) for offline training, VHDL for hardware inference

## Repository Structure

```
├── python/          # Offline training pipeline (snnTorch, WFDB, Quantization)
├── src/             # VHDL RTL source code (LIF neuron, Controller, FSM)
├── tb/              # VHDL testbenches and test vectors
├── constraints/     # Xilinx XDC constraint files
├── sim/             # Vivado simulation TCL scripts
└── docs/            # Project reports and block diagrams
```

## Workflow

1. **Offline Training (Python):** ECG signals are downloaded, preprocessed (R-peak detection, windowing), and used to train a PyTorch-based SNN using surrogate gradients.
2. **Quantization:** Trained weights are quantized to 8-bit fixed-point and exported as `.coe` / `.hex` files.
3. **Hardware Inference (VHDL):** The quantized weights are loaded into FPGA Block RAM. The VHDL inference engine rate-encodes the incoming ECG signal, processes it through the time-multiplexed hidden and output layers over 16 timesteps, and outputs the predicted class via an Argmax function.

## Tools Required
* **Hardware:** Xilinx Vivado Design Suite
* **Software:** Python 3.8+, PyTorch, snnTorch, WFDB
