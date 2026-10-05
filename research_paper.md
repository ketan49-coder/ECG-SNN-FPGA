# CardioSpike-FPGA: A Multiplier-Free Spiking Neural Network for Real-Time ECG Arrhythmia Classification on FPGA

---

**Authors:** Ketan Shinde, Rikhil Vaswami, Among R  
**Date:** September 2026

---

## Abstract

This paper presents **CardioSpike-FPGA**, a Spiking Neural Network designed for real-time ECG arrhythmia classification targeting FPGA deployment. The network classifies heartbeats from the MIT-BIH Arrhythmia Database into five AAMI classes using a 128→16→16→5 Leaky Integrate-and-Fire architecture that requires zero hardware multipliers. This paper covers the design rationale — from dataset analysis through preprocessing, architecture selection, training setup, and post-training quantization to 8-bit Q1.7 fixed-point — explaining how FPGA constraints shaped every decision in the software pipeline.

---

## 1. Introduction

A human heart beats roughly 100,000 times a day. Among those beats, a handful may be abnormal — a premature ventricular contraction, an atrial fibrillation episode, a fusion beat. Missing them can be fatal. The clinical gold standard is the Holter monitor: a patient wears it for 24–48 hours, and a cardiologist later reviews the recording. This workflow is slow, expensive, and reactive.

The question that started this project was simple: *what if the device on the patient's chest could classify every heartbeat in real time?* That requires an inference engine that is low-power, low-latency, and small enough for a wearable. Conventional deep neural networks fail on all three counts — their multiply-accumulate operations demand DSP slices, wide data paths, and significant dynamic power.

The alternative is a computational model that biology solved long ago: biological neurons do not multiply. They accumulate charge, and when it crosses a threshold, they fire a binary spike. The entire synaptic operation reduces to a conditional addition — if the presynaptic neuron fires, add the weight; if not, do nothing. The leak between spikes, if the decay factor is chosen carefully, becomes a bit-shift. No multiplier needed, anywhere.

This paper traces how that insight led to every design decision in our software pipeline — the architecture, the encoding, the training, the quantization — all shaped by the hardware that will eventually run it.

---

## 2. The Dataset and Its Challenges

The MIT-BIH Arrhythmia Database contains 48 half-hour recordings from 47 patients, sampled at 360 Hz, totaling 109,494 annotated beats across approximately 24.1 hours of continuous ECG data. We use only the MLII lead, which is present in 47 of the 48 records and provides the clearest R-peak morphology.

The AAMI standard groups the fine-grained annotations into five superclasses:

| AAMI Class | MIT-BIH Symbols | Count | Percentage |
|------------|-----------------|-------|------------|
| **N** (Normal) | N, L, R, e, j | 90,631 | **80.5%** |
| **S** (SVEB) | A, a, J, S | 2,781 | 2.5% |
| **V** (VEB) | V, E | 7,236 | 6.4% |
| **F** (Fusion) | F | 803 | **0.7%** |
| **Q** (Unknown) | /, f, Q | 8,043 | 7.1% |

The numbers reveal an immediate problem. 80.5% of all beats are Normal. If a network simply predicts "Normal" for every input, it achieves 80.5% accuracy — and misses every dangerous arrhythmia. The Fusion class has only 803 samples, 113× fewer than Normal. This imbalance shaped two later design decisions: inverse-frequency class weighting in the loss function, and targeted data augmentation.

---

## 3. Preprocessing

For each annotated R-peak, we extract a **128-sample window** centered on the peak. At 360 Hz, this spans 356 ms — enough to capture the QRS complex. But the real reason for 128 is architectural: 128 = 2⁷ maps to a 7-bit BRAM address, and the first-layer weight matrix (128 × 16 = 2,048 entries) tiles perfectly into standard BRAM blocks.

Each window is normalized to [0, 1] using min-max scaling. This maps directly to spike rate encoding: 1.0 means "fire every timestep," 0.0 means "never fire." Normalization is the bridge between the analog ECG world and the binary spike world.

After processing all 48 records, the pipeline yields ~100,000 heartbeat windows, split 80/20 with stratified sampling.

---

## 4. Augmentation

The class imbalance cannot be solved by preprocessing alone — 640 Fusion beats in the training set is not enough to learn the morphology. We apply four augmentation techniques, **only to minority classes**:

| Technique | What It Simulates | Parameters |
|-----------|-------------------|------------|
| Gaussian Noise | Electrode contact noise | σ = 0.02 |
| Time Shift | Imperfect R-peak detection | ±8 samples |
| Amplitude Scaling | Electrode placement variation | 0.85× to 1.15× |
| Baseline Wander | Respiratory artifact | Sinusoidal, max 0.05 |

All four are applied in sequence to each synthetic sample, clipped to [0, 1]. The target is to bring each minority class up to the count of the largest minority class — not to match Normal.

---

## 5. Architecture: 128 → 16 → 16 → 5

The architecture was not found by hyperparameter search. It was derived from FPGA constraints.

The first attempt was a single hidden layer of 32 neurons (128 → 32 → 5). It has 4,256 weights, requires a 5-bit neuron counter, and offers only one layer of feature abstraction — raw spike patterns mapped directly to class labels.

Splitting the hidden representation into **two layers of 16** changes three things. First, it creates a feature hierarchy: the first layer compresses 128 inputs into 16 features, the second layer combines them into class-discriminative representations. Two layers are more powerful than one, even with fewer neurons.

Second, 16 = 2⁴. A 4-bit counter handles time-multiplexing. The weight matrices (128×16, 16×16, 16×5) are all cleanly addressable.

Third, the total weight count drops to 2,048 + 256 + 80 = **2,384** — 44% fewer than the single-layer design, fitting in a single 18Kb BRAM block at 8 bits per weight.

| Architecture | Weights | BRAM | Multiplexer |
|-------------|---------|------|-------------|
| 128 → 32 → 5 | 4,256 | 4.16 KB | 5-bit |
| **128 → 16 → 16 → 5** | **2,384** | **2.34 KB** | **4-bit** |

---

## 6. LIF Neuron Parameters

The decay factor **β = 0.9375** = $1 - 2^{-4}$. In hardware, the leak becomes $V_{\text{mem}} - (V_{\text{mem}} \gg 4)$ — a shift and subtract, no multiplier.

The threshold is set to **0.5** instead of the default 1.0. In a three-layer SNN, a high threshold causes **vanishing spikes**: each layer fires so rarely that the output layer receives almost no input. A lower threshold keeps the hidden layers active enough for classification.

The reset is **hard reset to zero** — a MUX selecting zero after firing, the simplest possible hardware operation.

Spike encoding uses **rate coding** over **16 timesteps**. Each input value (0–1) is compared against an LFSR-generated random threshold at each step. Sixteen timesteps provide 17 distinguishable firing rates — sufficient for 5-class discrimination without excessive inference time.

---

## 7. Training Setup

Training uses **surrogate gradient descent** via snnTorch on Google Colab (GPU). The spike function is non-differentiable; during backpropagation, its gradient is replaced with a fast sigmoid surrogate (slope = 25).

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 5×10⁻³) |
| LR Schedule | StepLR, ×0.5 every 15 epochs |
| Batch Size | 128 |
| Epochs | 50 |
| Loss | Cross-Entropy, inverse-frequency weighted |
| Classification | Sum output spikes over 16 timesteps |

The class weighting penalizes minority-class errors heavily: Fusion (0.7% of data) receives ~28× the weight of Normal (80.5%). Without this, the network converges to >80% accuracy by predicting Normal for everything.

---

## 8. Quantization to Q1.7

After training, float32 weights are quantized to **8-bit Q1.7 fixed-point** — 1 sign bit, 7 fractional bits, range [-1.0, +0.9921875], resolution 0.0078125.

This format fits because trained SNN weights cluster in [-0.5, +0.5], and 8 bits = 1 byte = 1 BRAM word. The quantization pipeline clamps, scales by 128, rounds, and exports as Xilinx `.coe` files (for BRAM initialization) and `.hex` files (for testbench `$readmemh`).

A quantization error report documents per-layer max/mean/RMS error and clipped weight counts. The quantized model is evaluated on the test set to verify accuracy degradation stays below 1%.

---

## 9. The Road Ahead

The software pipeline — preprocessing, augmentation, training, and quantization — is implemented and ready to run. The next phases are:

1. **Train on Colab** and evaluate accuracy across all five AAMI classes.
2. **Quantize** the trained weights and verify minimal accuracy loss.
3. **Implement the VHDL inference engine** — time-multiplexed LIF neuron, weight BRAM, controller FSM, and Argmax output.
4. **Simulate and synthesize** on a Xilinx Artix-7 FPGA, targeting zero DSP utilization.

---

## References

1. Moody, G.B. & Mark, R.G. (2001). "The impact of the MIT-BIH Arrhythmia Database." *IEEE Eng. in Med. and Bio. Magazine*, 20(3), 45–50.
2. De Chazal, P. et al. (2004). "Automatic classification of heartbeats using ECG morphology and heartbeat interval features." *IEEE Trans. Biomed. Eng.*, 51(7), 1196–1206.
3. Kachuee, M. et al. (2018). "ECG Heartbeat Classification: A Deep Transferable Representation." *arXiv:1805.00794*.
4. Eshraghian, J.K. et al. (2023). "Training Spiking Neural Networks Using Lessons from Deep Learning." *Proc. IEEE*, 111(9), 1016–1054.
5. Neftci, E.O. et al. (2019). "Surrogate Gradient Learning in Spiking Neural Networks." *IEEE Signal Proc. Mag.*, 36(6), 51–63.
6. Hannun, A.Y. et al. (2019). "Cardiologist-level arrhythmia detection." *Nature Medicine*, 25, 65–69.
7. Xilinx Inc. (2020). "7 Series FPGAs Memory Resources User Guide." *UG473*.
