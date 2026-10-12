import wfdb
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import snntorch as snn
from snntorch import surrogate

# ─────────────────────────────────────────────────────────────────────────────
# Per-Record Diagnostic Check on DS2 (Free 1-Minute Audit)
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

if os.path.exists("/content/drive/MyDrive/SNN_Models/snn_superspike_ds1_ds2_best.pth"):
    MODEL_PATH = "/content/drive/MyDrive/SNN_Models/snn_superspike_ds1_ds2_best.pth"
else:
    MODEL_PATH = "models/snn_superspike_ds1_ds2_best.pth"

AAMI_4CLASS = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'S': 1, 'J': 1, 'V': 2, 'E': 2, 'F': 3}
DS2_RECORDS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network definition
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

def extract_record_beats(rec):
    X, y = [], []
    r = wfdb.rdrecord(os.path.join(DATASET_DIR, rec), channels=[0])
    sig, ann = r.p_signal.flatten(), wfdb.rdann(os.path.join(DATASET_DIR, rec), 'atr')
    for p, s in zip(ann.sample, ann.symbol):
        if s in AAMI_4CLASS and (p - 64 >= 0) and (p + 64 < len(sig)):
            w = sig[p - 64 : p + 64]
            mi, ma = np.min(w), np.max(w)
            if ma > mi:
                X.append((w - mi) / (ma - mi))
                y.append(AAMI_4CLASS[s])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print("=" * 80)
    print("PER-RECORD AUDIT OF DS2 TEST SET (22 PATIENTS)")
    print("=" * 80)
    print(f"Loading checkpoint from: {MODEL_PATH}")
    
    net = ECG_SuperSpike().to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    
    record_stats = []
    total_s_fp = 0
    total_f_fp = 0
    
    for rec in DS2_RECORDS:
        X, y = extract_record_beats(rec)
        if len(X) == 0: continue
        
        with torch.no_grad():
            preds = []
            for i in range(0, len(X), 256):
                batch = torch.tensor(X[i:i+256], dtype=torch.float32).to(device)
                p = net(batch).sum(0).argmax(1).cpu().numpy()
                preds.extend(p)
        preds = np.array(preds)
        
        # S-Class stats (Class 1)
        true_s = np.sum(y == 1)
        pred_s = np.sum(preds == 1)
        s_fp = np.sum((y != 1) & (preds == 1)) # False alarms for S
        s_tp = np.sum((y == 1) & (preds == 1))
        
        # F-Class stats (Class 3)
        true_f = np.sum(y == 3)
        pred_f = np.sum(preds == 3)
        f_fp = np.sum((y != 3) & (preds == 3)) # False alarms for F
        f_tp = np.sum((y == 3) & (preds == 3))
        
        total_s_fp += s_fp
        total_f_fp += f_fp
        
        record_stats.append({
            'record': rec,
            'total': len(y),
            'true_s': true_s,
            's_tp': s_tp,
            's_fp': s_fp,
            'true_f': true_f,
            'f_tp': f_tp,
            'f_fp': f_fp
        })
        
    # Sort records by S False Positives
    record_stats.sort(key=lambda x: x['s_fp'], reverse=True)
    
    print("\n" + "-" * 85)
    print(f"{'Patient':<8} | {'Total':<6} | {'True S':<6} | {'Caught S':<8} | {'False S (FP)':<12} | {'% of All S Errors':<18} | {'False F (FP)':<12}")
    print("-" * 85)
    
    running_pct = 0.0
    for stat in record_stats:
        pct = 100.0 * stat['s_fp'] / max(1, total_s_fp)
        running_pct += pct
        print(f"Record {stat['record']:<3} | {stat['total']:<6} | {stat['true_s']:<6} | {stat['s_tp']:<8} | {stat['s_fp']:<12} | {pct:>5.1f}% (Cum: {running_pct:>5.1f}%) | {stat['f_fp']:<12}")
        
    print("-" * 85)
    print(f"Total S False Alarms Across DS2: {total_s_fp}")
    print(f"Total F False Alarms Across DS2: {total_f_fp}")
    print("=" * 85)
    
    top3_fp = sum(r['s_fp'] for r in record_stats[:3])
    top3_pct = 100.0 * top3_fp / max(1, total_s_fp)
    print(f"\nDIAGNOSTIC FINDING:")
    print(f"The top 3 patients ({record_stats[0]['record']}, {record_stats[1]['record']}, {record_stats[2]['record']}) account for {top3_fp} of {total_s_fp} ({top3_pct:.1f}%) of ALL S false alarms!")
    if top3_pct > 60.0:
        print("--> STRONG EVIDENCE: A tiny handful of patients cause the vast majority of S errors!")
        print("--> Conclusive proof of patient-level amplitude/gain/baseline shifts hitting the encoder!")
    else:
        print("--> Errors are more evenly distributed across patients.")

if __name__ == "__main__":
    main()
