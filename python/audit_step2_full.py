import wfdb
import numpy as np
import os
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ─────────────────────────────────────────────────────────────────────────────
# Full 22-Patient Per-Record Audit for Step 2 (Robust Normalization Run)
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

MODEL_PATH = "/content/drive/MyDrive/SNN_Models/snn_superspike_ds1_ds2_step2_best.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/snn_superspike_ds1_ds2_step2_best.pth"

AAMI_4CLASS = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'S': 1, 'J': 1, 'V': 2, 'E': 2, 'F': 3}
DS2_RECORDS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ECG_SuperSpike(nn.Module):
    def __init__(self):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(128, 32, bias=False)
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

def extract_record_robust_norm(rec):
    X, y = [], []
    r = wfdb.rdrecord(os.path.join(DATASET_DIR, rec), channels=[0])
    sig = r.p_signal.flatten()
    ann = wfdb.rdann(os.path.join(DATASET_DIR, rec), 'atr')
    
    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) * 1.4826
    if mad < 1e-4: mad = np.std(sig) + 1e-4
    sig_std = (sig - med) / mad
    sig_norm = np.clip((sig_std + 4.0) / 14.0, 0.0, 1.0)
    
    for p, s in zip(ann.sample, ann.symbol):
        if s in AAMI_4CLASS and (p - 64 >= 0) and (p + 64 < len(sig)):
            X.append(sig_norm[p - 64 : p + 64])
            y.append(AAMI_4CLASS[s])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("=" * 105)
    print("FULL 22-PATIENT PER-RECORD AUDIT ON DS2: STEP 2 (ROBUST NORMALIZATION)")
    print("=" * 105)
    print(f"Loading checkpoint from: {MODEL_PATH}")
    
    net = ECG_SuperSpike().to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    
    stats = []
    tot_s_fp, tot_v_fp, tot_f_fp = 0, 0, 0
    tot_s_tp, tot_v_tp, tot_f_tp = 0, 0, 0
    tot_s_true, tot_v_true, tot_f_true = 0, 0, 0
    
    for rec in DS2_RECORDS:
        X, y = extract_record_robust_norm(rec)
        if len(X) == 0: continue
        
        with torch.no_grad():
            preds = []
            for i in range(0, len(X), 256):
                batch = torch.tensor(X[i:i+256], dtype=torch.float32).to(device)
                preds.extend(net(batch).sum(0).argmax(1).cpu().numpy())
        preds = np.array(preds)
        
        # S (1), V (2), F (3)
        s_true, s_tp, s_fp = np.sum(y == 1), np.sum((y == 1) & (preds == 1)), np.sum((y != 1) & (preds == 1))
        v_true, v_tp, v_fp = np.sum(y == 2), np.sum((y == 2) & (preds == 2)), np.sum((y != 2) & (preds == 2))
        f_true, f_tp, f_fp = np.sum(y == 3), np.sum((y == 3) & (preds == 3)), np.sum((y != 3) & (preds == 3))
        
        tot_s_true += s_true; tot_s_tp += s_tp; tot_s_fp += s_fp
        tot_v_true += v_true; tot_v_tp += v_tp; tot_v_fp += v_fp
        tot_f_true += f_true; tot_f_tp += f_tp; tot_f_fp += f_fp
        
        stats.append({
            'rec': rec, 'total': len(y),
            's_true': s_true, 's_tp': s_tp, 's_fp': s_fp,
            'v_true': v_true, 'v_tp': v_tp, 'v_fp': v_fp,
            'f_true': f_true, 'f_tp': f_tp, 'f_fp': f_fp
        })
        
    # Sort by False S descending
    stats.sort(key=lambda x: x['s_fp'], reverse=True)
    
    print("\n" + "-" * 105)
    print(f"{'Record':<7} | {'Total':<6} | {'True S':<7} | {'Caught S':<9} | {'False S (FP)':<13} | {'True V':<7} | {'Caught V':<9} | {'False V (FP)':<13} | {'False F (FP)':<12}")
    print("-" * 105)
    
    for s in stats:
        s_rec_str = f"{s['s_tp']} ({100.0*s['s_tp']/max(1, s['s_true']):.1f}%)" if s['s_true'] > 0 else "0 (N/A)"
        v_rec_str = f"{s['v_tp']} ({100.0*s['v_tp']/max(1, s['v_true']):.1f}%)" if s['v_true'] > 0 else "0 (N/A)"
        print(f"Rec {s['rec']:<4} | {s['total']:<6} | {s['s_true']:<7} | {s_rec_str:<9} | {s['s_fp']:<13} | {s['v_true']:<7} | {v_rec_str:<9} | {s['v_fp']:<13} | {s['f_fp']:<12}")
        
    print("-" * 105)
    print(f"TOTALS  | {sum(s['total'] for s in stats):<6} | {tot_s_true:<7} | {tot_s_tp} ({100.0*tot_s_tp/tot_s_true:.1f}%) | {tot_s_fp:<13} | {tot_v_true:<7} | {tot_v_tp} ({100.0*tot_v_tp/tot_v_true:.1f}%) | {tot_v_fp:<13} | {tot_f_fp:<12}")
    print("=" * 105)

if __name__ == "__main__":
    main()
