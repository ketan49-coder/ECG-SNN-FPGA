# Comprehensive Evaluation Report: Intra-Patient vs. De Chazal Inter-Patient Benchmark

**Project:** Multiplier-Free Spiking Neural Network on Spartan-7 FPGA for ECG Arrhythmia Classification  
**Date:** October 11, 2026  
**Status:** Baseline Finalized & Evaluated across Dual Benchmarks  

---

## 1. Executive Summary & Comparative Results

This report documents the performance of the 32-neuron multiplier-free SuperSpike SNN across two distinct evaluation methodologies: the conventional **80/20 Stratified Random Split** and the canonical **de Chazal (2004) / AAMI EC57 Inter-Patient Split (DS1/DS2)**.

### Comparative Benchmark Summary
| Evaluation Protocol | Classes Evaluated | Preprocessing / Partitioning Protocol | Model Precision | Overall Accuracy | Macro-F1 Score | Test Set Size ($N$) | Primary Hardware Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Stratified Random Split (80/20)** | **5 Classes** (N, S, V, F, Q) | Beat-level stratified sampling across 48 records. Individual patient morphology present in train and test. | Float32: 95.19%<br>**Q1.7 Quantized: 95.34%** | **95.34%** (Quantized) | **80.10%** | 21,896 beats | **Target for FPGA Demonstration** (`snn_pkg_superspike_q17.vhd`) |
| **De Chazal Inter-Patient Split (DS1/DS2)** | **4 Classes** (N, S, V, F) | 44 records (4 paced records dropped). Record-level partition: DS1 (Train/Val), DS2 (Test). **No patient overlap.** | Float32: 81.77% | **81.77%** | **38.16%** | 49,696 beats | Benchmark for Scientific Discussion & Future Ablation |

> [!NOTE]
> **Class Set Discrepancy Note:** The random 80/20 split evaluates 5 classes because the 4 paced records (`102`, `104`, `107`, `217`) were included, generating 1,609 paced/unclassifiable beats mapped to Class Q. In the official de Chazal DS1/DS2 benchmark, standard AAMI EC57 protocol mandates excluding paced records. Because non-paced records contain fewer than 0.02% Class Q beats (only 7 beats in DS2), Class Q was dropped, establishing the canonical 4-class problem (N, S, V, F).

---

## 2. Benchmark 1: Stratified Random Split (80/20) — Hardware Baseline

This model serves as the core baseline for physical FPGA verification. It was trained for 60 epochs with Van Rossum distance loss ($\tau = 2.0$), Cosine Annealing learning rate scheduling, and evaluated under both Float32 and strict 8-bit Q1.7 post-training quantization.

### Classification Report (8-Bit Q1.7 Quantized Model)
* **Overall Accuracy:** 95.34% *(Float32: 95.52% $\to$ Quantization drop: only -0.18%)*
* **Macro-F1 Score:** 80.10%

| Class | Precision | Recall | F1-Score | Support ($N$) | Clinical & Diagnostic Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal (N)** | 0.9798 | 0.9721 | 0.9759 | 18,124 | Dominant class; represents 82.8% of test set ($18,124 / 21,896$). |
| **Supraventricular (S)** | 0.5378 | 0.7032 | 0.6095 | 556 | Handled via augmentation, but shows persistent 46.2% false alarm rate. |
| **Ventricular (V)** | 0.8679 | 0.8535 | 0.8606 | 1,447 | Strong balance; 1,235 true ventricular arrhythmias captured. |
| **Fusion (F)** | 0.4804 | 0.8438 | 0.6122 | 160 | High recall (84.4%), but 48.0% precision; small sample size ($N=160$). |
| **Unknown / Paced (Q)** | 0.9872 | 0.9099 | 0.9470 | 1,609 | High metrics driven by sharp, artificial pacemaker spikes. |

### Confusion Matrix (Random 80/20 Split)
![Confusion Matrix](confudion matrix.png)

```
              Predicted N   Predicted S   Predicted V   Predicted F   Predicted Q
True N:          17,618         287           126            82            11
True S:             146         391            10             3             6
True V:             130          28         1,235            52             2
True F:              20           1             4           135             0
True Q:              68          20            48             9         1,464
```

---

## 3. Benchmark 2: De Chazal Inter-Patient Split (DS1 / DS2) — Scientific Benchmark

To evaluate true generalization to unseen patient physiology, the network was retrained strictly using the de Chazal (2004) partition with **no patient overlap between train, validation, and test sets**.
* **DS1 Training:** 18 records (augmented to 96,048 samples).
* **DS1 Validation:** 4 records (`209`, `215`, `220`, `223`).
* **DS2 Independent Testing:** 22 records (49,696 raw beats).

### Classification Report (DS2 Test Set — 22 Unseen Patients)
* **Overall Accuracy:** 81.77%
* **Macro-F1 Score:** 38.16%

| Class | Precision | Recall | F1-Score | Support ($N$) | Clinical Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal (N)** | 0.9380 | 0.8807 | 0.9085 | 44,251 | Solid baseline specificity (88.1% recall across 44k beats). |
| **Supraventricular (S)** | **0.0057** | **0.0114** | **0.0076** | 1,837 | **Severe Collapse:** Only 21 of 1,837 beats caught; 1,795 called Normal. |
| **Ventricular (V)** | 0.4457 | 0.4789 | 0.4617 | 3,220 | Moderate generalization; 1,542 true V beats caught in unseen hearts. |
| **Fusion (F)** | 0.1036 | 0.2629 | 0.1486 | 388 | Modest sensitivity; 102 true fusion beats caught. |

### Confusion Matrix (DS2 Test Set — 49,696 Beats)
```
              Predicted N   Predicted S   Predicted V   Predicted F
True N:          38,973        2,865         1,894           519
True S:           1,795           21            16             5
True V:             505          814         1,542           359
True F:             274            4             8           102
```

---

## 4. Deep Analysis: The Supraventricular (S) Failure

### A. Why Random Splits Mask Failure (The Methodological Contribution)
Under the random 80/20 split, Class S achieved **70.32% recall**. Under the de Chazal split, S recall collapsed to **1.14%**. 

This stark contrast provides a crucial methodological finding for the biomedical neuromorphic community: **random beat-level splits create an illusion of generalization by allowing the network to memorize patient-specific ectopic morphology.** When evaluated across unseen patients, morphology-only networks fail completely on supraventricular ectopy.

### B. The Clinical & Algorithmic Etiology
1. **Absence of Temporal Context (The RR Hypothesis):** Electrophysiologically, a Premature Atrial Contraction (PAC / Class S) originates above the bundle of His. Consequently, ventricular conduction proceeds along the normal His-Purkinje pathway, producing a QRS complex whose morphology is essentially indistinguishable from a normal sinus beat in a single lead. Cardiologists diagnose PACs by identifying **prematurity** (a shortened preceding RR interval). Because our current SNN receives only an isolated 128-sample waveform window, it is deprived of all RR-interval timing context.
   * *Paper Phrasing:* **Morphology-only input likely cannot separate S from N across patients; RR-interval features are a standard remedy in the literature and represent the planned next experiment.**
2. **Sub-Chance Performance (The Systematic Bias):** Class S precision on DS2 was **0.57%**, predicting S for 2,865 normal beats and 814 ventricular beats while missing 1,816 true S beats. Because 0.57% is substantially below the test base rate (3.7%), this failure is not merely random confusion with Normal beats; it reveals a systematic distortion.
   * *Open Investigation:* The augmentation pipeline oversampled from only 391 raw DS1 S-beats, likely constructing an over-fitted synthetic prototype from 1–2 training patients. Additionally, patient-to-patient baseline and gain variations may shift features across the fixed rate-encoding threshold. These represent open experimental questions to be addressed through per-record normalization and class-weighted loss functions.

---

## 5. Paper Organization & Disclosures

### A. How to Structure the Paper
* **Results Section:** Present both evaluations in a single comparative table. Provide full per-class metrics and confusion matrices for both runs to showcase transparent, thorough benchmarking.
* **Discussion & Limitations:** 
  * Explicitly discuss the Class S collapse and the 4-class vs. 5-class discrepancy.
  * Disclose the unstable validation training curve (where peak validation occurred at Epoch 1 due to majority-class bias when using accuracy-based selection).
* **Future Work Section:** Formalize the planned ablation sequence: (1) Checkpoint selection on validation Macro-F1 with 7 validation records, (2) Robust per-record normalization, (3) Multi-peak RR-interval counter integration, and (4) On-chip continuous learning via DCLL.

### B. Mandatory Hardware Disclosure
> [!IMPORTANT]
> **FPGA Deployment Caveat:** The hardware implementation and FPGA demonstration (`snn_pkg_superspike_q17.vhd` running on the Xilinx Spartan-7) uses the **random-split model weights (95.34% quantized accuracy)**. The hardware demonstration rigorously proves that the multiplier-free, bit-shift LIF architecture executes with high throughput and near-zero quantization loss on silicon, but it does **not** carry the inter-patient generalization claim.
