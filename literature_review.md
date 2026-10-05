# Literature Review & Key References
## ECG-SNN-FPGA Project: Spiking Neural Network for ECG Arrhythmia Classification on FPGA

This curated list of 20 references is structured exactly how you should present them in your final paper or defense.

---

## 🏆 Top 4 Core Benchmark & Foundation Papers
*These are the most important papers for your project. B1 is your direct competitor, Zenke is your mathematical foundation, B2 is your CNN benchmark, and A3 validates your hardware approach.*

### 1. [Main Competitor - Must Beat] All-Spiking ECG Analysis for Arrhythmia Classification on Low-Power FPGA
- **Authors:** Matteo Antonio Scrugli, et al.
- **Year:** 2026 | **Journal:** *IEEE Sensors Journal*
- **DOI:** [10.1109/JSEN.2026.3663715](https://doi.org/10.1109/JSEN.2026.3663715)
- **Why it matters:** This is our #1 target. They achieved **98.4% accuracy** on MIT-BIH using an SNN on a Lattice FPGA. We compare our accuracy and hardware efficiency directly against them.

### 2. [Theoretical Foundation] Surrogate Gradient Learning in Spiking Neural Networks
- **Authors:** Emre O. Neftci, Hesham Mostafa, Friedemann Zenke
- **Year:** 2019 | **Journal:** *IEEE Signal Processing Magazine*
- **DOI:** [10.1109/MSP.2019.2931595](https://doi.org/10.1109/MSP.2019.2931595)
- **Why it matters:** This is the math behind snnTorch. Because spikes (0 or 1) are not differentiable, standard backpropagation fails. Zenke's surrogate gradient method solves this, allowing us to train SNNs in Python.

### 3. [CNN Benchmark - To Beat on Power] FPGA-Based Real-Time ECG Classification System Using Quantized Inception-ResNeXt
- **Authors:** Tiancheng Cao, et al.
- **Year:** 2026 | **Journal:** *IEEE TVLSI*
- **DOI:** [10.1109/TVLSI.2024.3472270](https://doi.org/10.1109/TVLSI.2024.3472270)
- **Why it matters:** They achieved a massive **99.5% accuracy**, but they used a heavy CNN with DSP multipliers. We use them to prove our Pareto tradeoff: we trade a tiny bit of accuracy to save massive amounts of hardware power by using a multiplier-free SNN.

### 4. [Hardware Validation] An Approximate Computing-Based SNN Neuron Model
- **Authors:** Haihang Xia, et al.
- **Year:** 2026 | **Journal:** *IEEE TCAS-I*
- **DOI:** [10.1109/TCSI.2025.3624352](https://doi.org/10.1109/TCSI.2025.3624352)
- **Why it matters:** Validates our multiplier-free LIF approach. They showed that dropping DSP multipliers boosts energy efficiency by 6.75×.

---

## 🧠 Group 1: Identical / Highly Similar SNN Hardware Approaches

### 5. A Neuromorphic Processing System With Spike-Driven SNN Processor for Wearable ECG Classification
- **Authors:** Hongwei Chu, et al. (2022)
- **Relevance:** An event-driven SNN on FPGA for ECG. We contrast our synchronous FSM approach with their asynchronous approach.

### 6. An Optimized Multi-Layer SNN Implementation in FPGA Without Multipliers
- **Authors:** Ali Mehrabi, et al. (2023)
- **Relevance:** Uses adder trees and bit-shifts to bypass DSPs, practically identical to our VHDL architecture.

### 7. QCSNN: A Memory-Efficient SNN for On-Device ECG Arrhythmia Detection
- **Authors:** Olamilekan Banjo, Behnaz Ghoraani (2025)
- **Relevance:** Quantized LIF neurons on a PYNQ-Z2 FPGA for MIT-BIH classification. Validates the viability of quantized LIFs for edge ECG.

### 8. Ultra-Energy-Efficient ECG Classification Processor with SNN Inference
- **Authors:** Ruixin Mao, et al. (2022)
- **Relevance:** Ultra-low-power SNN inference processor achieving sub-microjoule energy per beat.

---

## ⚙️ Group 2: Justifying Our FPGA Architecture Decisions

### 9. A Low-Cost High-Speed Neuromorphic Hardware Based on SNN
- **Authors:** Edris Zaman Farsa, et al. (2019)
- **Relevance:** Further academic proof that shift-and-add logic successfully replaces multipliers in neuromorphic hardware.

### 10. Toward the Optimal Design and FPGA Implementation of SNNs
- **Authors:** Wenzhe Guo, et al. (2022)
- **Relevance:** Proves that **Q8.8 fixed-point arithmetic** (which we use) is the optimal tradeoff between accuracy and LUT usage on Xilinx FPGAs.

### 11. Low-Energy and Fast SNN for Context-Dependent Learning on FPGA
- **Authors:** Hajar Asgari, et al. (2020)
- **Relevance:** Validates running digital LIF accelerators specifically on **Xilinx Spartan/Artix** series FPGAs using rate coding.

### 12. A Generalized Hardware Architecture for Real-Time SNNs
- **Authors:** Daniel Valencia, et al. (2023)
- **Relevance:** Validates our FSM controller approach: processing one neuron at a time (time-multiplexed) saves massive amounts of FPGA area.

### 13. Synaptic Activity and Hardware Footprint of SNNs in Digital Neuromorphic Systems
- **Authors:** Edgar Lemaire, et al. (2022)
- **Relevance:** Quantitative analysis of fixed-point precision requirements and memory bottlenecks in FPGA SNNs.

### 14. Reconstruction of a Fully Paralleled Auditory SNN and FPGA Implementation
- **Authors:** Bin Deng, et al. (2021)
- **Relevance:** Demonstrates Xilinx 7-series (Spartan-7/Artix-7) mapping and fixed-point math for bio-inspired neural networks.

---

## 📊 Group 3: Justifying Our Python Data Pipeline

### 15. ECG Heartbeat Classification: A Deep Transferable Representation [⭐ Landmark]
- **Authors:** Mohammad Kachuee, et al. (2018)
- **Relevance:** The foundational paper that established the 5-class AAMI mapping (N, S, V, F, Q) on the MIT-BIH dataset. Every paper cites this.

### 16. Generalization of CNNs for ECG Classification Using GANs
- **Authors:** Abdelrahman M. Shaker, et al. (2020)
- **Relevance:** Proves that class imbalance is the #1 problem in MIT-BIH and requires data augmentation (justifying our `augment_data.py`).

### 17. A Robust Deep CNN with Batch-Weighted Loss for Heartbeat Classification
- **Authors:** Ali Sellami, et al. (2019)
- **Relevance:** Validates our use of `class_weights` in the PyTorch loss function to penalize missing rare arrhythmias.

### 18. ECGTransForm: Adaptive ECG Arrhythmia Classification
- **Authors:** Hany El-Ghaish, et al. (2024)
- **Relevance:** Modern paper showing that context-aware weighted loss functions are still the standard for this dataset.

### 19. Transfer Learning for ECG Classification
- **Authors:** Kuba Weimann, Tim O. F. Conrad (2021)
- **Relevance:** Standardizes the bandpass filtering, resampling, and heartbeat normalization pipeline across patient records.

---

## 🚀 Group 4: Context and "Future Work" Justifications

### 20. SyncNN: Evaluating and Accelerating SNNs on FPGAs
- **Authors:** Sathish Panchapakesan, et al. (2020)
- **Relevance:** Compares rate coding vs. temporal coding. We cite this to defend our choice of rate coding for its hardware simplicity.

---
*Note: A1 (Spiker+) and B9 (STDP Wearables) have been explicitly excluded from this list for strategic focus, reserving them for future solo publication work.*
