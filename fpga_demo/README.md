# 🫀 Neuromorphic SNN ECG Hardware Live Demonstration

### 4-Class Arrhythmia Classifier on Xilinx Spartan-7 (XC7S15)

---

## 🚀 Quick Start for Live Presentation (Rikhil's Laptop)

### 1. Requirements
Ensure Python is installed on your laptop, then open Command Prompt or Terminal:
```bash
pip install pyserial matplotlib numpy
```

### 2. Launch the Application
Navigate to this folder and run:
```bash
python ecg_fpga_gui.py
```

---

## 🖥️ How the Demo Works

1. **Before the FPGA is Connected (Simulation / Dry Run Mode):**
   * The app will open in **Software Digital Twin Mode**.
   * You can select any patient heartbeat from the dropdown (Normal, Supraventricular, Ventricular Arrhythmia, Fusion).
   * Click **`⚡ TRANSMIT & CLASSIFY`** to preview the ECG waveform and see the diagnostic output and simulated 4 ms latency.

2. **When the Physical FPGA Board is Plugged In via USB:**
   * Plug the FPGA board's USB cable into the laptop.
   * Click the **`🔄`** refresh button next to the COM port dropdown.
   * Select the FPGA's COM port (e.g., `COM3`, `COM4`, `/dev/ttyUSB0`).
   * Click **`Connect to FPGA`** (turns Green).
   * Select a heartbeat and click **`⚡ TRANSMIT & CLASSIFY`**:
     * The laptop transmits 128 ECG bytes across the USB cable at 115,200 baud.
     * The physical Spartan-7 FPGA runs the multiplier-free SNN.
     * The FPGA illuminates the on-board LED corresponding to the diagnosis!
     * The FPGA replies back to the laptop in **4.08 milliseconds** with the diagnosis banner!

---

## 📊 Hardware Specifications (Talking Points for Professors)

| Metric | Measured Value | Significance |
| :--- | :---: | :--- |
| **FPGA Target** | Xilinx Spartan-7 (`XC7S15`) | Smallest, lowest-cost 28nm FPGA |
| **DSP Multipliers** | **0 DSP48E1 Slices** | 100% Multiplier-Free Shift-and-Subtract |
| **Logic Footprint** | **1,318 LUTs (16.48%)** | Tiny silicon area, fits in low-cost chips |
| **Block RAM** | **0 BRAM Blocks** (24 LUTRAMs) | Extremely fast distributed ROM |
| **Operating Power** | **30 mW** baseline | Ideal for battery-powered wearable ECGs |
| **Inference Latency** | **4.08 ms** ($T=75$ timesteps) | **196x faster** than human heartbeat ($800\text{ ms}$) |
