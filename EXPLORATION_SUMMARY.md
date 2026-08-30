# MIT-BIH Arrhythmia Database - Data Exploration Summary

## Dataset Overview

| Property | Value |
|----------|-------|
| **Total Records** | 48 (main) + 23 (supplementary x_mitdb) |
| **Sampling Rate** | 360 Hz |
| **Duration per Record** | ~30.1 minutes (650,000 samples) |
| **Total Duration** | ~24.1 hours (main) + ~13.8 hours (supplementary) |
| **Leads Available** | MLII, V1, V2, V4, V5 (2 leads per record) |
| **Total Annotated Beats** | 109,494 (main database) |
| **Signal Units** | mV (millivolts) |
| **ADC Gain** | 200 |

## AAMI Class Distribution (Main Database)

| Class | Symbol | Count | Percentage | Description |
|-------|--------|-------|------------|-------------|
| **N** | N, L, R, e, j | 90,631 | 80.5% | Normal beats |
| **S** | A, a, J, S | 2,781 | 2.5% | Supraventricular ectopic beats (SVEB) |
| **V** | V, E | 7,236 | 6.4% | Ventricular ectopic beats (VEB) |
| **F** | F | 803 | 0.7% | Fusion beats |
| **Q** | /, f, Q | 8,043 | 7.1% | Unknown/Paced beats |
| **Non-beat** | +, ~, \|, [, ], !, ", x | 3,153 | 2.8% | Non-beat annotations |

### Key Observations:
- **Highly imbalanced**: Normal beats dominate (80.5%)
- **VEB class**: 7,236 beats across multiple records - sufficient for training
- **SVEB class**: Only 2,781 beats - more challenging for classification
- **Unknown class (Q)**: 7.1% - many paced beats in records 102, 104, 107, 217

## Lead Configuration

| Lead | Records | Notes |
|------|---------|-------|
| MLII | 47/48 | Present in almost all records (modified Lead II) |
| V1 | 28/48 | Most common precordial lead |
| V2 | 5/48 | Records 102, 103, 104, 117, 123 |
| V4 | 1/48 | Record 124 only |
| V5 | 8/48 | Records 100, 102, 103, 104, 114, 123 |

**Standard configuration**: MLII + V1 (used in 32 records)

## Records with Highest Arrhythmia Content

### Top 5 by VEB (Ventricular Ectopic Beats):
1. **Record 208**: 992 VEBs (33.6%) - MLII + V1
2. **Record 233**: 831 VEBs (27.0%) - MLII + V1
3. **Record 200**: 826 VEBs (31.8%) - MLII + V1
4. **Record 106**: 520 VEBs (25.7%) - MLII + V1
5. **Record 223**: 473 VEBs (18.2%) - MLII + V1

### Top 5 by SVEB (Supraventricular Ectopic Beats):
1. **Record 232**: 1,382 SVEBs (77.6%) - MLII + V1 (atrial fibrillation)
2. **Record 209**: 383 SVEBs (12.7%) - MLII + V1
3. **Record 222**: 209 SVEBs (8.4%) - MLII + V1
4. **Record 201**: 128 SVEBs (6.5%) - MLII + V1
5. **Record 207**: 107 SVEBs (5.8%) - MLII + V1

### Records Dominated by VEBs (>10%):
- 208, 233, 200, 106, 223, 231, 214, 228, 217, 221, 203, 213, 124, 210, 119, 116, 105, 109

### Records with Atrial Fibrillation:
- **232**: 77.6% SVEB (atrial fibrillation)
- **209**: 12.7% SVEB
- **222**: 8.4% SVEB

## Signal Characteristics

### Signal Statistics (MLII lead):
- **Mean**: -0.2 to -0.8 mV (varies by record)
- **Std Dev**: 0.15 to 0.86 mV
- **Range**: 1.3 to 10.2 mV
- Records 102, 103, 107, 108 show highest variability

### RR Interval Analysis (47 records):
- **Mean RR Interval**: ~800-1200 ms (varies significantly)
- **Mean Heart Rate**: 45-100 bpm (highly variable due to arrhythmias)
- **RR Variability (CV)**: Highly variable (records with arrhythmias show higher CV)

## Supplementary Database (x_mitdb)

- **23 records**: x_108 through x_234 (matching main records)
- **Shorter duration**: 10 minutes each (216,000 samples)
- **Same leads**: MLII + V1
- **Purpose**: Additional annotations for specific records

## Generated Visualizations

| File | Description |
|------|-------------|
| `dashboard.png` | Comprehensive 12-panel overview |
| `aami_distribution.png` | Overall AAMI class distribution bar chart |
| `aami_pie.png` | Pie chart of main beat classes |
| `per_record_aami.png` | Stacked bar chart per record |
| `signal_samples.png` | 8 records × first 5 seconds with beat annotations |
| `beat_morphology.png` | Average beat morphology (Normal, VEB, SVEB) for 3 records |
| `beat_comparison.png` | Overlay comparison of Normal vs VEB across records |
| `signal_heatmap.png` | Signal mean/std/range heatmap by record and lead |
| `rr_analysis.png` | RR interval statistics per record |
| `duration_vs_beats.png` | Duration vs beat count scatter |
| `signal_stats.csv` | Signal statistics for all records/leads |
| `rr_intervals.csv` | RR interval statistics per record |
| `aami_distribution.json` | Complete AAMI class counts per record |

## Recommendations for ML/DL Applications

### Train/Test Split Strategy:
1. **Record-wise split** (not beat-wise) to avoid data leakage
2. **Standard splits** (AAMI recommended):
   - DS1: Records 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
   - DS2: Records 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

### Preprocessing Considerations:
1. **Bandpass filter**: 0.5-40 Hz recommended
2. **Normalization**: Per-record or global z-score
3. **Beat segmentation**: 240ms window (90ms pre-R, 150ms post-R)
4. **Class balancing**: Oversampling/undersampling or weighted loss for VEB/SVEB
5. **Lead selection**: MLII is most consistent; consider single-lead or multi-lead approaches

### Class Mapping for Training:
```
N, L, R, e, j → Class 0 (Normal)
A, a, J, S    → Class 1 (SVEB)
V, E          → Class 2 (VEB)
F             → Class 3 (Fusion)
/, f, Q       → Class 4 (Unknown/Paced) - often excluded
```

## Files Generated

```
SNN_ECG/
├── EXPLORATION_REPORT.txt      # Full text report
├── EXPLORATION_SUMMARY.md      # This markdown summary
├── aami_distribution.json      # Complete AAMI class counts
├── signal_stats.csv            # Signal statistics per record/lead
├── rr_intervals.csv            # RR interval statistics per record
├── create_dashboard.py         # Visualization script (in dataset folder)
├── create_dashboard_main.py    # Visualization script (in main folder)
└── Visualizations:
    ├── dashboard.png
    ├── aami_distribution.png
    ├── aami_pie.png
    ├── per_record_aami.png
    ├── signal_samples.png
    ├── beat_morphology.png
    ├── beat_comparison.png
    ├── signal_heatmap.png
    ├── rr_analysis.png
    └── duration_vs_beats.png
```

---
*Exploration completed using Python with wfdb, numpy, pandas, matplotlib, seaborn*
*All exploration files are in the main SNN_ECG folder (not in the dataset folder)*