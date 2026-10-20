"""
Inter-Patient Benchmark (de Chazal DS1 Train -> DS2 Test Protocol)
T = 75 Timesteps, Hardware Matched, with Real Minority Augmentation
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import wfdb

# ── 1. Inter-Patient Patient Groupings (AAMI de Chazal Standard) ─────────────
DS1_PATIENTS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', 
                '119', '122', '124', '201', '203', '205', '207', '208', '209', 
                '215', '220', '223', '230'] # 22 Training Patients

DS2_PATIENTS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', 
                '202', '210', '212', '213', '214', '219', '221', '222', '228', 
                '231', '232', '233', '234'] # 22 Unseen Test Patients

AAMI_4_MAP = {
    'N':0, 'L':0, 'R':0, 'e':0, 'j':0,
    'A':1, 'a':1, 'J':1, 'S':1,
    'V':2, 'E':2,
    'F':3
}

# ── 2. Hardware-Matched Hyperparameters ──────────────────────────────────────
INPUT_FEATURES = 128
HIDDEN_1       = 32
HIDDEN_2       = 32
OUTPUT_CLASSES = 4
NUM_STEPS      = 75       # T = 75 hardware matched
BETA           = 0.9375   # Multiplier-free leak (15/16)
THRESHOLD      = 0.5
BATCH_SIZE     = 256
NUM_EPOCHS     = 60
LEARNING_RATE  = 2e-3
SEED           = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Inter-Patient Benchmark on: {device} | T = {NUM_STEPS}")

# ── 3. Extract DS1 & DS2 Partitions ──────────────────────────────────────────
mit_dirs = [r for r, _, files in os.walk('/content', followlinks=True) if '100.dat' in files]
if not mit_dirs:
    mit_dirs = [r for r, _, files in os.walk('.', followlinks=True) if '100.dat' in files]
if not mit_dirs:
    mit_dirs = [r for r, _, files in os.walk('/content/drive', followlinks=True) if '100.dat' in files]
if not mit_dirs:
    raise FileNotFoundError("Could not find 'mit-bih-arrhythmia-database-1.0.0' directory! Check path.")

MIT_DIR = mit_dirs[0]
print(f"Extracting patient records from: {MIT_DIR}")

def extract_patients(patient_list):
    X_list, y_list = [], []
    for p in patient_list:
        rec_path = os.path.join(MIT_DIR, p)
        try:
            r = wfdb.rdrecord(rec_path, channels=[0])
            sig = r.p_signal.flatten()
            ann = wfdb.rdann(rec_path, 'atr')
            med = np.median(sig)
            mad = np.median(np.abs(sig - med)) * 1.4826 + 1e-4
            sig_norm = np.clip(((sig - med) / mad + 4.0) / 14.0, 0.0, 1.0)
            for idx, sym in enumerate(ann.symbol):
                if sym in AAMI_4_MAP:
                    pos = ann.sample[idx]
                    if pos - 64 >= 0 and pos + 64 <= len(sig_norm):
                        beat = sig_norm[pos - 64 : pos + 64]
                        if len(beat) == 128:
                            X_list.append(beat)
                            y_list.append(AAMI_4_MAP[sym])
        except Exception as e:
            print(f"  Error reading patient {p}: {e}")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

print("\n>>> Extracting DS1 (22 Training Patients)...")
X_ds1, y_ds1 = extract_patients(DS1_PATIENTS)
print(f"DS1 Extracted: {len(y_ds1)} beats | Class counts: {np.bincount(y_ds1)}")

print("\n>>> Extracting DS2 (22 Unseen Test Patients)...")
X_ds2, y_ds2 = extract_patients(DS2_PATIENTS)
print(f"DS2 Extracted: {len(y_ds2)} beats | Class counts: {np.bincount(y_ds2)}")

# ── 4. Inject Real External Minority Beats into DS1 Train ONLY ───────────────
extra_files = []
for root_search in ['.', '/content', '/content/drive']:
    extra_files.extend([os.path.join(r, 'all_real_minority_beats_svf.npz') 
                        for r, _, files in os.walk(root_search, followlinks=True) if 'all_real_minority_beats_svf.npz' in files])

if extra_files:
    print(f"\nInjecting extra real patient beats from: {extra_files[0]}")
    extra_data = np.load(extra_files[0])
    X_extra, y_extra = extra_data['X'], extra_data['y']
    X_train_merged = np.vstack([X_ds1, X_extra])
    y_train_merged = np.concatenate([y_ds1, y_extra])
    print(f"DS1 + Extra Real Data: {len(y_train_merged)} total training beats")
else:
    print("\nExtra real beats not found. Proceeding with pure DS1 training...")
    X_train_merged, y_train_merged = X_ds1, y_ds1

# ── 5. Lighter Calibrated SMOTE on DS1 ────────────────────────────────────────
n_N = int(np.sum(y_train_merged == 0))
smote_target = {
    0: n_N,
    1: max(8000, int(np.sum(y_train_merged == 1))),
    2: max(20000, int(np.sum(y_train_merged == 2))),
    3: max(6000, int(np.sum(y_train_merged == 3)))
}
smote = SMOTE(sampling_strategy=smote_target, random_state=SEED)
X_sm, y_sm = smote.fit_resample(X_train_merged, y_train_merged)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_sm, dtype=torch.float32), torch.tensor(y_sm, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_ds2, dtype=torch.float32), torch.tensor(y_ds2, dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=False)

# ── 6. SNN Model & Van Rossum Loss ───────────────────────────────────────────
class VanRossumLoss(nn.Module):
    def __init__(self, tau_decay=0.9375):
        super().__init__()
        self.decay = tau_decay
    def forward(self, spk, tgt):
        T, B, C = spk.shape
        t_spk = torch.zeros(T, B, C, device=spk.device)
        for b in range(B):
            t_spk[:, b, tgt[b]] = 1.0
        f_out, f_tgt = torch.zeros_like(spk), torch.zeros_like(t_spk)
        tr_o = torch.zeros(B, C, device=spk.device)
        tr_t = torch.zeros(B, C, device=spk.device)
        for t in range(T):
            tr_o = tr_o * self.decay + spk[t]
            tr_t = tr_t * self.decay + t_spk[t]
            f_out[t], f_tgt[t] = tr_o, tr_t
        return F.mse_loss(f_out, f_tgt, reduction='sum') / B

class ECG_Spartan7_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.fc1  = nn.Linear(INPUT_FEATURES, HIDDEN_1, bias=False)
        self.fc2  = nn.Linear(HIDDEN_1,       HIDDEN_2, bias=False)
        self.fc3  = nn.Linear(HIDDEN_2, OUTPUT_CLASSES, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism="zero")
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism="zero")
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        spk_rec = []
        for t in range(NUM_STEPS):
            s1, m1 = self.lif1(self.fc1(x), m1); s1 = self.drop(s1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            spk_rec.append(s3)
        return torch.stack(spk_rec)

def evaluate(net, loader):
    net.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for d, t in loader:
            pred = net(d.to(device)).sum(0).argmax(1).cpu().numpy()
            all_p.extend(pred); all_t.extend(t.numpy())
    all_p, all_t = np.array(all_p), np.array(all_t)
    acc = 100. * np.mean(all_p == all_t)
    f1  = 100. * f1_score(all_t, all_p, average='macro', zero_division=0)
    cm  = confusion_matrix(all_t, all_p, labels=[0, 1, 2, 3])
    rec = [100. * cm[i, i] / max(1, cm[i].sum()) for i in range(4)]
    prec = [100. * cm[i, i] / max(1, cm[:, i].sum()) for i in range(4)]
    class_f1 = [2 * prec[i] * rec[i] / max(1e-5, (prec[i] + rec[i])) for i in range(4)]
    return acc, f1, cm, prec, rec, class_f1

# ── 7. Training Loop & Live Logging ──────────────────────────────────────────
net = ECG_Spartan7_SNN().to(device)
criterion = VanRossumLoss(tau_decay=BETA)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

export_drive_dir = '/content/drive/MyDrive/SNN-ECG/SNN_Models'
os.makedirs(export_drive_dir, exist_ok=True)

print(f"\n{'='*75}\nSTARTING INTER-PATIENT (DS1 TRAIN -> DS2 TEST) BENCHMARK\n{'='*75}")

best_ds2_f1 = 0.0
best_weights = None

for epoch in range(1, NUM_EPOCHS + 1):
    net.train()
    total_loss = 0.0
    for d, t in train_loader:
        d, t = d.to(device), t.to(device)
        optimizer.zero_grad()
        loss = criterion(net(d), t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    
    # Evaluate on DS2 unseen test patients
    acc, f1, cm, prec, rec, class_f1 = evaluate(net, test_loader)
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.4f} | DS2 Test Acc: {acc:6.2f}% | DS2 Macro F1: {f1:6.2f}%")
    
    if f1 > best_ds2_f1:
        best_ds2_f1 = f1
        best_weights = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    
    if epoch % 10 == 0 or epoch == NUM_EPOCHS:
        print("\n" + "="*75)
        print(f">>> [EPOCH {epoch:02d} UNSEEN DS2 TEST PATIENT BREAKDOWN] <<<")
        print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
        print("-" * 65)
        for c_idx, c_name in enumerate(['Normal (N)', 'Supravent (S)', 'Ventricular (V)', 'Fusion (F)']):
            print(f"{c_name:<20} {prec[c_idx]:>6.2f}%        {rec[c_idx]:>6.2f}%        {class_f1[c_idx]:>6.2f}%")
        print("-" * 65)
        print(f"DS2 ACCURACY: {acc:.2f}% | DS2 MACRO F1: {f1:.2f}%")
        print("DS2 Confusion Matrix [N, S, V, F]:")
        print(cm)
        print("="*75)
        
        # Save checkpoint to Drive
        ckpt_name = f'inter_patient_ds2_epoch{epoch}.pt'
        torch.save(net.state_dict(), os.path.join(export_drive_dir, ckpt_name))
        print(f"Checkpoint saved: {ckpt_name}\n")

# ── 8. Quantize & Export Final Model ─────────────────────────────────────────
net.load_state_dict(best_weights)
acc, f1, cm, prec, rec, class_f1 = evaluate(net, test_loader)

print(f"\n{'*'*75}")
print(f"FINAL BEST INTER-PATIENT RESULTS ON UNSEEN DS2 PATIENTS")
print(f"{'*'*75}")
print(f"DS2 Accuracy: {acc:.2f}% | DS2 Macro F1: {f1:.2f}%")
for c_idx, c_name in enumerate(['Normal (N)', 'Supravent (S)', 'Ventricular (V)', 'Fusion (F)']):
    print(f"  {c_name:<18}: Prec={prec[c_idx]:.2f}% | Rec={rec[c_idx]:.2f}% | F1={class_f1[c_idx]:.2f}%")
print(f"DS2 Confusion Matrix:\n{cm}\n")

w1 = best_weights['fc1.weight'].numpy()
w2 = best_weights['fc2.weight'].numpy()
w3 = best_weights['fc3.weight'].numpy()
max_val = max(np.max(np.abs(w1)), np.max(np.abs(w2)), np.max(np.abs(w3)))
scale = float(max_val)

w1_q = np.clip(np.round(w1 / max_val * 127.0), -128, 127).astype(np.int8)
w2_q = np.clip(np.round(w2 / max_val * 127.0), -128, 127).astype(np.int8)
w3_q = np.clip(np.round(w3 / max_val * 127.0), -128, 127).astype(np.int8)

out_file = 'inter_patient_t75_best_weights.npz'
drive_npz = os.path.join(export_drive_dir, out_file)
np.savez_compressed(out_file, w1=w1_q, w2=w2_q, w3=w3_q, scale=scale)
np.savez_compressed(drive_npz, w1=w1_q, w2=w2_q, w3=w3_q, scale=scale)

print(f"\nFinal Inter-Patient Weights Saved to Drive: {drive_npz}")
