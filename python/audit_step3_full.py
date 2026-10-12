import wfdb
import numpy as np
import os
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ─────────────────────────────────────────────────────────────────────────────
# Full 22-Patient Per-Record Audit for Step 3 (RR Features + Robust Normalization)
# ─────────────────────────────────────────────────────────────────────────────
POSSIBLE_PATHS = [
    r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0",
    r"mit-bih-arrhythmia-database-1.0.0",
    r"/content/drive/MyDrive/SNN-ECG/mit-bih-arrhythmia-database-1.0.0",
    r"/content/mit-bih-arrhythmia-database-1.0.0"
]

DATASET_DIR = None
for root, dirs, files in os.walk('/content'):
    if '100.hea' in files or '100.dat' in files:
        DATASET_DIR = root
        break

if DATASET_DIR is None:
    for p in POSSIBLE_PATHS:
        if os.path.exists(p) and (os.path.exists(os.path.join(p, "100.dat")) or os.path.exists(os.path.join(p, "100.hea"))):
            DATASET_DIR = p
            break

if DATASET_DIR is None:
    DATASET_DIR = r"mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"

MODEL_PATH = "/content/drive/MyDrive/SNN_Models/snn_superspike_ds1_ds2_step3_best.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/snn_superspike_ds1_ds2_step3_best.pth"

AAMI_4CLASS = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'S': 1, 'J': 1, 'V': 2, 'E': 2, 'F': 3}
DS2_RECORDS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ECG_SuperSpike(nn.Module):
    def __init__(self):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(131, 32, bias=False)
        self.fc2 = nn.Linear(32, 32, bias=False)
        self.fc3 = nn.Linear(32, 4, bias=False)
        self.lif1 = snn.Leaky(beta=0.9375, threshold=0.5, spike_grad=grad, reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta=0.9375, threshold=0.5, spike_grad=grad, reset_mechanism="zero")
        self.lif3 = snn.Leaky(beta=0.9375, threshold=0.5, spike_grad=grad, reset_mechanism="zero")

    def forward(self, x):
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        spk_rec = []
        for t in range(25):
            s1, m1 = self.lif1(self.fc1(x), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            spk_rec.append(s3)
        return torch.stack(spk_rec)

def extract_record_step3(rec):
    X, y = [], []
    r = wfdb.rdrecord(os.path.join(DATASET_DIR, rec), channels=[0])
    sig = r.p_signal.flatten()
    ann = wfdb.rdann(os.path.join(DATASET_DIR, rec), 'atr')
    
    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) * 1.4826
    if mad < 1e-4: mad = np.std(sig) + 1e-4
    sig_std = (sig - med) / mad
    sig_norm = np.clip((sig_std + 4.0) / 14.0, 0.0, 1.0)
    
    samples = ann.sample
    symbols = ann.symbol
    n_beats = len(samples)
    history_rrs = []
    
    for i in range(n_beats):
        p, s = samples[i], symbols[i]
        if s not in AAMI_4CLASS or (p - 64 < 0) or (p + 64 >= len(sig)):
            continue
            
        rr_pre = (samples[i] - samples[i-1]) / 360.0 if i > 0 else (samples[1] - samples[0]) / 360.0
        rr_post = (samples[i+1] - samples[i]) / 360.0 if i < n_beats - 1 else rr_pre
        
        history_rrs.append(rr_pre)
        if len(history_rrs) > 10:
            history_rrs.pop(0)
        local_mean = np.mean(history_rrs)
        rr_ratio = rr_pre / (local_mean + 1e-4)
        
        rr_pre_feat = np.clip(rr_pre / 2.0, 0.0, 1.0)
        rr_post_feat = np.clip(rr_post / 2.0, 0.0, 1.0)
        rr_ratio_feat = np.clip(rr_ratio / 2.0, 0.0, 1.0)
        
        feat = np.hstack([sig_norm[p-64:p+64], [rr_pre_feat, rr_post_feat, rr_ratio_feat]])
        X.append(feat)
        y.append(AAMI_4CLASS[s])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("=" * 110)
    print("FULL 22-PATIENT PER-RECORD AUDIT ON DS2: STEP 3 (ROBUST NORMALIZATION + 3 RR FEATURES)")
    print("=" * 110)
    print(f"Loading checkpoint from: {MODEL_PATH}")
    
    net = ECG_SuperSpike().to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    
    stats = []
    tot_s_fp, tot_v_fp, tot_f_fp = 0, 0, 0
    tot_s_tp, tot_v_tp, tot_f_tp = 0, 0, 0
    tot_s_true, tot_v_true, tot_f_true = 0, 0, 0
    
    for rec in DS2_RECORDS:
        X, y = extract_record_step3(rec)
        if len(X) == 0: continue
        
        with torch.no_grad():
            preds = net(torch.tensor(X, dtype=torch.float32).to(device)).sum(0).argmax(1).cpu().numpy()
            
        s_true = int(np.sum(y == 1)); s_pred = int(np.sum(preds == 1))
        s_tp   = int(np.sum((y == 1) & (preds == 1)))
        s_fp   = int(np.sum((y != 1) & (preds == 1)))
        s_rec  = 100.0 * s_tp / s_true if s_true > 0 else 0.0
        
        v_true = int(np.sum(y == 2)); v_pred = int(np.sum(preds == 2))
        v_tp   = int(np.sum((y == 2) & (preds == 2)))
        v_fp   = int(np.sum((y != 2) & (preds == 2)))
        v_rec  = 100.0 * v_tp / v_true if v_true > 0 else 0.0

        f_true = int(np.sum(y == 3)); f_pred = int(np.sum(preds == 3))
        f_tp   = int(np.sum((y == 3) & (preds == 3)))
        f_fp   = int(np.sum((y != 3) & (preds == 3)))
        f_rec  = 100.0 * f_tp / f_true if f_true > 0 else 0.0

        acc    = 100.0 * np.sum(preds == y) / len(y)
        
        tot_s_fp += s_fp; tot_s_tp += s_tp; tot_s_true += s_true
        tot_v_fp += v_fp; tot_v_tp += v_tp; tot_v_true += v_true
        tot_f_fp += f_fp; tot_f_tp += f_tp; tot_f_true += f_true

        stats.append({
            'rec': rec, 'beats': len(y), 'acc': acc,
            's_true': s_true, 's_tp': s_tp, 's_fp': s_fp, 's_rec': s_rec,
            'v_true': v_true, 'v_tp': v_tp, 'v_fp': v_fp, 'v_rec': v_rec,
            'f_true': f_true, 'f_tp': f_tp, 'f_fp': f_fp, 'f_rec': f_rec,
        })
        
    print("\n| Patient | Total Beats | Overall Acc | True S | Caught S | S Recall | False S Alarms | True V | Caught V | V Recall | False V Alarms | True F | Caught F | F Recall | False F Alarms |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for s in stats:
        print(f"| {s['rec']} | {s['beats']} | {s['acc']:.1f}% | {s['s_true']} | {s['s_tp']} | {s['s_rec']:.1f}% | {s['s_fp']} | {s['v_true']} | {s['v_tp']} | {s['v_rec']:.1f}% | {s['v_fp']} | {s['f_true']} | {s['f_tp']} | {s['f_rec']:.1f}% | {s['f_fp']} |")
        
    print("-" * 110)
    print(f"POOLED DS2 TOTALS:")
    print(f"  S (Supraventricular): True={tot_s_true}, Caught={tot_s_tp} ({100.0*tot_s_tp/tot_s_true:.2f}% Recall), False Alarms={tot_s_fp}")
    print(f"  V (Ventricular):      True={tot_v_true}, Caught={tot_v_tp} ({100.0*tot_v_tp/tot_v_true:.2f}% Recall), False Alarms={tot_v_fp}")
    print(f"  F (Fusion):           True={tot_f_true}, Caught={tot_f_tp} ({100.0*tot_f_tp/tot_f_true:.2f}% Recall), False Alarms={tot_f_fp}")
    print("=" * 110)

if __name__ == "__main__":
    main()
