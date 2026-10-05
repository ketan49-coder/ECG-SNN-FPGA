# Project Proposal

**DES Pune University**  
**Department of Electronics and Communication Engineering (AIML)**  
Academic Year: 2025–2026

---

**Project Title:** CardioSpike-FPGA: A Multiplier-Free Spiking Neural Network for On-Device ECG Arrhythmia Classification

**Team Members:** Ketan Shinde, Rikhil Vaswami, Among R  
**Guide:** Dr. Deepali Newaskar

---

## Acknowledgment

We would like to sincerely thank our project guide, Dr. Deepali Newaskar, for their continuous guidance, constructive feedback, and encouragement throughout the development of this project. We are grateful to the Department of Electronics and Communication Engineering (AIML) at DES Pune University for providing us with the resources and academic environment necessary to pursue this work. We also extend our thanks to the open-source communities behind the MIT-BIH Arrhythmia Database (PhysioNet) and the snnTorch library, whose publicly available tools made this research possible.

---

## 1. Introduction

Cardiac arrhythmias are abnormal heart rhythms that represent one of the leading causes of sudden cardiac death and hospitalization globally. Early and continuous detection of arrhythmias can significantly improve patient outcomes, particularly for high-risk individuals who require round-the-clock cardiac monitoring outside of a clinical setting.

Traditional clinical ECG monitors are large, power-hungry devices confined to hospital environments. While wearable ECG devices exist, most rely on sending raw data to a cloud server or smartphone for processing, introducing latency and dependence on wireless connectivity. True on-device, real-time arrhythmia classification — where the computation happens directly on a low-power chip worn by the patient — remains a largely unsolved engineering challenge.

Spiking Neural Networks (SNNs) offer a biologically inspired computational paradigm ideally suited for this challenge. Unlike standard Artificial Neural Networks (ANNs), SNNs process information as discrete binary spike events. Because a spike is simply a 1 or a 0, multiplications are replaced by conditional additions — a property that maps directly and efficiently onto digital hardware fabric such as Field-Programmable Gate Arrays (FPGAs).

This project, **CardioSpike-FPGA**, proposes the design and implementation of a fully digital, multiplier-free SNN-based ECG classifier on a standard commodity FPGA. The entire inference pipeline — from raw ECG input to arrhythmia class output — runs on the FPGA chip with zero dependence on an external processor or cloud.

---

## 2. Problem Statement

Current state-of-the-art approaches to on-device ECG classification face one or more of the following critical limitations:

1. **Excessive Power Consumption:** Recent high-accuracy neural network implementations (e.g., Dual-Head Quantized Convolutional SNNs on PYNQ-Z2) consume upward of 330 milliwatts of power. A standard wearable medical patch powered by a CR2032 coin-cell battery would be drained in under two hours under such a load, making these solutions impractical for long-term patient monitoring.

2. **Reliance on SoC-Class Hardware:** Several published FPGA implementations require System-on-Chip (SoC) platforms that embed full ARM Cortex processors (e.g., Xilinx Zynq-7020). These platforms are expensive, physically large, and consume orders of magnitude more power than bare FPGA fabric.

3. **Hardware Multiplier Dependency:** Many works claiming to be "SNN-based" rely on hybrid ANN-SNN architectures or convolutional operations that still require DSP slices (dedicated hardware multipliers) on the FPGA. This defeats the primary hardware efficiency advantage of spiking computation.

4. **Complex Pre-Processing Overhead:** Certain implementations shift the computational burden from the neural network to the input encoding stage, requiring real-time computation of mathematical derivatives of the ECG signal in hardware, adding latency, logic gates, and power consumption.

**CardioSpike-FPGA** addresses all four of these limitations simultaneously through a carefully co-designed software training pipeline and hardware architecture.

---

## 3. Objectives

The primary objectives of this project are as follows:

1. **To design a 3-layer Leaky Integrate-and-Fire (LIF) Spiking Neural Network** with the topology 128 → 16 → 16 → 5, trained on the MIT-BIH Arrhythmia Database to classify the five AAMI standard heartbeat categories: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), and Unknown (Q).

2. **To implement a hardware-aware training pipeline** in Python (using the snnTorch library) that enforces a membrane leak rate of exactly β = 0.9375, mathematically guaranteeing that the LIF update reduces to a single arithmetic right-shift operation (`>> 4`) in hardware, eliminating all multipliers.

3. **To quantize all trained network weights** to the Q1.7 fixed-point format (8-bit signed, 1 sign bit + 7 fractional bits), enabling storage of all 2,384 network weights within a single 18Kb Block RAM (BRAM) tile on the FPGA.

4. **To implement the complete SNN inference engine in VHDL**, synthesizable on standard commodity FPGAs (Xilinx Artix-7 / Spartan-7 family), utilizing zero DSP slices.

5. **To validate the hardware implementation** against the software model, confirming classification correctness and measuring FPGA resource utilization (LUT count, BRAM usage, power estimate).

---

## 4. Brief Literature Review

The field of on-device neural ECG classification has grown significantly in recent years. The following works are most directly relevant to this project:

**Scrugli et al. (DATE 2024) — "On-FPGA Spiking Neural Networks for Integrated Near-Sensor ECG Analysis"** [DOI: 10.23919/DATE58400.2024.10546736]: This foundational paper implements an SNN on the ultra-low-power Lattice iCE40-UltraPlus FPGA using delta-modulation spike encoding. While highly power-efficient, the approach requires real-time hardware computation of the first and second mathematical derivatives of the ECG signal as part of the input encoder, increasing pre-processing logic complexity. CardioSpike-FPGA eliminates this pre-processing overhead entirely by using direct-amplitude rate coding.

**"A Cascaded Quantized Spiking Neural Network for Real-Time ECG Arrhythmia Detection on Edge Hardware" (Sensors/MDPI, 2026)** [DOI: 10.3390/s26123723]: This work achieves 99.02% accuracy using a Dual-Head Quantized Convolutional SNN on a PYNQ-Z2 board. However, the platform requires an embedded ARM Cortex-A9 processor and consumes 330 mW, disqualifying it from coin-cell wearable applications. CardioSpike-FPGA targets bare FPGA fabric without any embedded processor.

**Kohno Lab, University of Tokyo (Frontiers in Neuroscience, 2014)** [DOI: 10.3389/fnins.2014.00396]: Introduces the Digital Spiking Silicon Neuron (DSSN) model on FPGA for high-fidelity biological simulation. The DSSN model prioritizes biological realism over hardware efficiency, requiring complex piecewise-linear approximations of ion channel dynamics. CardioSpike-FPGA instead adopts the mathematically simplest viable neuron model (LIF), minimizing per-neuron logic gate cost.

**"A Systematic Review of ECG Arrhythmia Classification" (arXiv, 2025)** [arXiv:2503.07276]: A comprehensive survey of 2017–2024 ECG classification literature, highlighting the systemic problem of class imbalance in the MIT-BIH dataset (Normal beats constitute over 80% of samples). This motivates CardioSpike-FPGA's use of class-weighted loss functions and data augmentation during training.

---

## 5. Proposed Methodology

### 5.1 Dataset
The **MIT-BIH Arrhythmia Database** (PhysioNet) is used as the primary dataset. It contains 48 half-hour, two-lead ECG recordings sampled at 360 Hz from 47 patients, with over 109,000 annotated heartbeats. Annotations are remapped to the five AAMI standard classes (N, S, V, F, Q) for classification.

### 5.2 Pre-processing Pipeline
1. R-peak locations are extracted from the database annotations.
2. A 128-sample window (approximately 355 ms at 360 Hz) is centered on each R-peak.
3. Each window is normalized to the [0, 1] range using min-max normalization.
4. An 80/20 stratified train/test split is applied, preserving class proportions.
5. Minority classes (S, V, F, Q) are augmented using Gaussian noise injection, amplitude scaling, and time-shifting to address class imbalance.

### 5.3 Network Architecture
The SNN architecture is a 3-layer fully connected network:

| Layer | Input | Output | Weights |
|---|---|---|---|
| FC1 + LIF1 | 128 | 16 | 2,048 |
| FC2 + LIF2 | 16 | 16 | 256 |
| FC3 + LIF3 | 16 | 5 | 80 |
| **Total** | | | **2,384** |

All neurons use the Leaky Integrate-and-Fire (LIF) model with:
- Leak rate: β = 0.9375 (maps to `>> 4` right-shift in hardware)
- Threshold: θ = 0.5
- Reset mechanism: Zero reset (subtractive)
- No bias terms (hardware simplification)

### 5.4 Training
- Framework: Python with snnTorch library on Google Colab (GPU-accelerated)
- Optimizer: Adam (lr = 5×10⁻³)
- Loss: Cross-Entropy with inverse-frequency class weights
- Epochs: 30, with learning rate step decay
- Time steps: 16 (network unrolled across 16 time steps per heartbeat)

### 5.5 Quantization
Trained weights are post-training quantized to **Q1.7 fixed-point format**:
- 1 sign bit + 7 fractional bits
- Range: [−1.0, +0.9921875]
- Resolution: 1/128 ≈ 0.0078
- All 2,384 weights fit within a single 18Kb FPGA BRAM tile
- Exported as Xilinx `.coe` files for BRAM initialization in Vivado

### 5.6 FPGA Implementation
The inference engine is implemented in VHDL with the following modules:
- **Weight BRAM:** Single 18Kb tile storing all Q1.7 quantized weights
- **MAC Unit:** Multiplier-free. Performs conditional addition (add weight if spike = 1, skip if spike = 0)
- **LIF Neuron:** Arithmetic right-shift for leak (`>> 4`), comparator for threshold firing
- **Controller FSM:** Time-multiplexed sequencing of all 16 time steps and 3 layers
- **Output Decoder:** Argmax of spike-count accumulators across 16 time steps

---

## 6. Applications

1. **Wearable Cardiac Patch:** The primary application is a long-term ambulatory ECG monitor worn as a chest patch. The ultra-low power budget of a bare FPGA implementation enables weeks of continuous operation on a small battery, suitable for post-surgery patients or high-risk individuals.

2. **Implantable Medical Devices:** Future miniaturization of this architecture could enable integration into pacemakers or implantable loop recorders, providing continuous arrhythmia monitoring without external hardware.

3. **Emergency Response Wearables:** First responders or military personnel could wear the device to provide real-time cardiac alerts under high-stress conditions where hospital monitoring is not feasible.

4. **Remote Patient Monitoring (Rural Healthcare):** In areas without reliable internet connectivity, on-device classification eliminates the need for cloud data transmission, making cardiac monitoring accessible in low-resource settings.

5. **Sports and Athlete Monitoring:** Athletes require real-time cardiac feedback during high-intensity training. The zero-latency, on-chip classification enables immediate arrhythmia alerts without the lag of cloud-based processing.

---

## 7. References

1. M. A. Scrugli et al., "On-FPGA Spiking Neural Networks for Integrated Near-Sensor ECG Analysis," in *Proc. DATE 2024*, DOI: 10.23919/DATE58400.2024.10546736

2. [Authors], "A Cascaded Quantized Spiking Neural Network for Real-Time ECG Arrhythmia Detection on Edge Hardware," *Sensors*, MDPI, 2026, DOI: 10.3390/s26123723

3. T. Kohno et al., "An FPGA-Based Silicon Neuronal Network with Selectable Excitability Silicon Neurons," *Frontiers in Neuroscience*, 2014, DOI: 10.3389/fnins.2014.00396

4. [Authors], "A Systematic Review of ECG Arrhythmia Classification: Adherence to Standards, Fair Evaluation, and Embedded Feasibility," *arXiv*, 2025, arXiv:2503.07276

5. G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, 2001.

6. J. Eshraghian et al., "Training Spiking Neural Networks Using Lessons from Deep Learning," *Proceedings of the IEEE*, 2023.
