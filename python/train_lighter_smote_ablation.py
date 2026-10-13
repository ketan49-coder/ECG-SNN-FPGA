"""
Lighter SMOTE Ablation — Seed 42 Only (Comparison Row, NOT Headline)
=====================================================================
S=8k, F=6k instead of S=25k, F=20k  |  T=25, 128 inputs, 60/20/20 split
Purpose: Test if reduced minority-class synthetic support raises S & F precision
         at the cost of slightly lower S & F recall, keeping F1 within ~3 points.

Pre-registered prediction (computed from Seed 42 enhanced confusion matrix):
  - S precision:  59.6%  → estimated 68–74%  (+)
  - F precision:  55.8%  → estimated 62–70%  (+)
  - S recall:     78.8%  → estimated 68–75%  (-)
  - F recall:     83.9%  → estimated 72–80%  (-)
  - S F1:         67.8   → stays within 3 pts (0.65–0.72)
  - F F1:         67.0   → stays within 3 pts (0.64–0.71)
  - Overall Acc:  96.45% → ~95.5–96.0%       (slight dip acceptable)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import wfdb

# ── Hyperparams (identical to enhanced benchmark) ─────────────────────────────
INPUT_FEATURES  = 128
HIDDEN_1        = 32
HIDDEN_2        = 32
OUTPUT_CLASSES  = 4
NUM_STEPS       = 25        # T=25, same as enhanced benchmark
BETA            = 0.9375    # Multiplier-free bit-shift leak
THRESHOLD       = 0.5
BATCH_SIZE      = 256
NUM_EPOCHS      = 60
LEARNING_RATE   = 2e-3
SEED            = 42        # Single seed, comparison row only

AAMI_4_MAP = {
    'N':0, 'L':0, 'R':0, 'e':0, 'j':0,
    'A':1, 'a':1, 'J':1, 'S':1,
    'V':2, 'E':2,
    'F':3
}
EXCLUDED_PACED = {'102','104','107','217'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load / build beat cache ────────────────────────────────────────────────────
cache_path = "mitbih_sparrow_4class_128.npz"
if os.path.exists(cache_path):
    cached = np.load(cache_path)
    X_all, y_all = cached['X'], cached['y']
    print(f"Loaded {len(y_all)} beats from cache.")
else:
    mit_candidates = [r for r, _, f in os.walk('/content', followlinks=True) if '100.dat' in f]
    MIT_DIR = mit_candidates[0] if mit_candidates else 'mitdb'
    print(f"Extracting from {MIT_DIR}...")
    all_recs = sorted(list(set([
        f.split('.')[0] for f in os.listdir(MIT_DIR)
        if f.endswith('.dat') and f.split('.')[0] not in EXCLUDED_PACED
    ])))
    X_list, y_list = [], []
    for rec in all_recs:
        p_rec = os.path.join(MIT_DIR, rec)
        try:
            r   = wfdb.rdrecord(p_rec, channels=[0])
            sig = r.p_signal.flatten()
            ann = wfdb.rdann(p_rec, 'atr')
            med = np.median(sig)
            mad = np.median(np.abs(sig - med)) * 1.4826 + 1e-4
            sig_norm = np.clip(((sig - med) / mad + 4.0) / 14.0, 0.0, 1.0)
            for i in range(len(ann.sample)):
                p, s = ann.sample[i], ann.symbol[i]
                if s not in AAMI_4_MAP or (p-64 < 0) or (p+64 > len(sig_norm)):
                    continue
                beat = sig_norm[p-64 : p+64]
                if len(beat) == 128:
                    X_list.append(beat); y_list.append(AAMI_4_MAP[s])
        except Exception as e:
            print(f"  Warning on {rec}: {e}")
    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.int64)
    np.savez_compressed(cache_path, X=X_all, y=y_all)
    print(f"Extracted and cached {len(y_all)} beats.")

# ── Augmentation (identical) ──────────────────────────────────────────────────
def augment_ecg_batch(x):
    scale = torch.empty(x.shape[0], 1, device=x.device).uniform_(0.90, 1.10)
    noise = torch.randn_like(x) * 0.01
    return torch.clamp(x * scale + noise, 0.0, 1.0)

# ── Van Rossum Loss (identical) ───────────────────────────────────────────────
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
        tr_o = tr_t = torch.zeros(B, C, device=spk.device)
        for t in range(T):
            tr_o = tr_o * self.decay + spk[t]
            tr_t = tr_t * self.decay + t_spk[t]
            f_out[t], f_tgt[t] = tr_o, tr_t
        return F.mse_loss(f_out, f_tgt, reduction='sum') / B

# ── SNN Architecture (identical) ─────────────────────────────────────────────
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
            s1, m1 = self.lif1(self.fc1(x), m1);  s1 = self.drop(s1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            spk_rec.append(s3)
        return torch.stack(spk_rec)

def evaluate(net, loader):
    net.eval(); all_p, all_t = [], []
    with torch.no_grad():
        for d, t in loader:
            d = d.to(device)
            pred = net(d).sum(0).argmax(1).cpu().numpy()
            all_p.extend(pred); all_t.extend(t.numpy())
    all_p, all_t = np.array(all_p), np.array(all_t)
    acc = 100. * np.mean(all_p == all_t)
    f1  = 100. * f1_score(all_t, all_p, average='macro', zero_division=0)
    return acc, f1, all_t, all_p

# ── 60/20/20 Split — IDENTICAL to enhanced benchmark, same seed ───────────────
torch.manual_seed(SEED); np.random.seed(SEED)
X_tv, X_test, y_tv, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv)
print(f"Partition: Train={len(y_train)} (60%) | Val={len(y_val)} (20%) | Test={len(y_test)} (20%)")

# ── LIGHTER Calibrated SMOTE — the ONLY change from the enhanced benchmark ────
n_N = int(np.sum(y_train == 0))
lighter_strategy = {
    0: n_N,    # Normal: natural count — no upsampling
    1: 8000,   # S:  8,000  (was 25,000 → 3x lighter)
    2: 25000,  # V: 25,000  (unchanged)
    3: 6000    # F:  6,000  (was 20,000 → 3.3x lighter)
}
smote = SMOTE(sampling_strategy=lighter_strategy, random_state=SEED)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
sm_counts = np.bincount(y_sm, minlength=4)

# ── PRINT CLASS COUNTS — confirm ratio changed ────────────────────────────────
print("\n" + "="*60)
print("SMOTE CLASS COUNTS AFTER RESAMPLING (verify ratio changed)")
print("="*60)
print(f"  Normal (N)       : {sm_counts[0]:>7}   (natural, no upsample)")
print(f"  Supravent (S)    : {sm_counts[1]:>7}   (target: 8,000 — was 25,000)")
print(f"  Ventricular (V)  : {sm_counts[2]:>7}   (target: 25,000 — unchanged)")
print(f"  Fusion (F)       : {sm_counts[3]:>7}   (target: 6,000 — was 20,000)")
print(f"  Total train      : {len(y_sm):>7}")
print("="*60 + "\n")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_sm,    dtype=torch.float32),
                  torch.tensor(y_sm,    dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                  torch.tensor(y_val,   dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test,  dtype=torch.float32),
                  torch.tensor(y_test,  dtype=torch.long)),
    batch_size=BATCH_SIZE, shuffle=False)

# ── Checkpoint directory ──────────────────────────────────────────────────────
CKPT_DIR = "/content/drive/MyDrive/SNN-ECG/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Checkpoints will be saved to: {CKPT_DIR}")

# ── Training loop ─────────────────────────────────────────────────────────────
net       = ECG_Spartan7_SNN().to(device)
criterion = VanRossumLoss(tau_decay=BETA)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_f1, best_state, best_epoch = 0.0, None, 0

for ep in range(1, NUM_EPOCHS + 1):
    net.train(); total_loss = 0.0
    for d, t in train_loader:
        d, t = d.to(device), t.to(device)
        d_aug = augment_ecg_batch(d)
        optimizer.zero_grad()
        loss = criterion(net(d_aug), t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    val_acc, val_f1, _, _ = evaluate(net, val_loader)
    marker = ""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1; best_epoch = ep
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        ckpt_path = os.path.join(CKPT_DIR, f"lighter_smote_seed{SEED}_best.pth")
        torch.save({'epoch': ep, 'val_f1': val_f1, 'state_dict': best_state}, ckpt_path)
        marker = " (*BEST VAL F1* — saved to Drive)"
    if ep % 5 == 0 or ep == 1 or marker:
        print(f"  [Ep {ep:02d}/{NUM_EPOCHS}] Loss: {total_loss/len(train_loader):.4f} "
              f"| Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}%{marker}")

# ── Evaluate best checkpoint on un-augmented test set ─────────────────────────
print(f"\nBest checkpoint: Epoch {best_epoch}, Val F1 {best_val_f1:.2f}%")
net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
t_acc, t_f1, y_true, y_pred = evaluate(net, test_loader)
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
labels_str = ['N','S','V','F']

# ── Per-class stats ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LIGHTER SMOTE ABLATION RESULTS — SEED 42 (COMPARISON ROW ONLY)")
print("="*70)
print(f"Overall Accuracy : {t_acc:.2f}%")
print(f"Macro-F1 Score   : {t_f1:.2f}%\n")
print(f"{'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>10}")
print("-"*50)
per_class = {}
for i, cls in enumerate(labels_str):
    TP = cm[i,i]; FP = cm[:,i].sum()-TP; FN = cm[i,:].sum()-TP
    prec = TP/(TP+FP+1e-9)*100
    rec  = TP/(TP+FN+1e-9)*100
    f1c  = 2*prec*rec/(prec+rec+1e-9)
    support = cm[i,:].sum()
    per_class[cls] = {'prec':prec,'rec':rec,'f1':f1c}
    print(f"  {cls:<4} {prec:>9.2f}%  {rec:>9.2f}%  {f1c:>7.2f}%  {support:>9}")

print("\nConfusion Matrix:")
print(f"         Pred N   Pred S   Pred V   Pred F")
for i, cls in enumerate(labels_str):
    print(f"True {cls}:  {'  '.join(f'{cm[i,j]:6}' for j in range(4))}")

# ── Delta vs enhanced benchmark (Seed 42, S=25k/F=20k) ───────────────────────
base = {
    'acc':96.45, 'f1':81.49,
    'prec_S':59.56, 'rec_S':78.75, 'f1_S':67.8,
    'prec_F':55.79, 'rec_F':83.85, 'f1_F':67.0
}
print("\n" + "="*70)
print("DELTA vs ENHANCED BENCHMARK (Seed 42  |  S=25k, F=20k)")
print("="*70)
def delta(new, old): return f"{old:.2f}% → {new:.2f}%   Δ={new-old:+.2f}%"
print(f"  Overall Acc  :  {delta(t_acc,            base['acc'])}")
print(f"  Macro-F1     :  {delta(t_f1,             base['f1'])}")
print(f"  S Precision  :  {delta(per_class['S']['prec'], base['prec_S'])}")
print(f"  S Recall     :  {delta(per_class['S']['rec'],  base['rec_S'])}")
print(f"  S F1         :  {delta(per_class['S']['f1'],   base['f1_S'])}")
print(f"  F Precision  :  {delta(per_class['F']['prec'], base['prec_F'])}")
print(f"  F Recall     :  {delta(per_class['F']['rec'],  base['rec_F'])}")
print(f"  F F1         :  {delta(per_class['F']['f1'],   base['f1_F'])}")
print("="*70)
print("\n[PREDICTION CHECK]")
print(f"  S prec >= 65%?  {'✅ PASS' if per_class['S']['prec'] >= 65 else '❌ MISS'}")
print(f"  F prec >= 60%?  {'✅ PASS' if per_class['F']['prec'] >= 60 else '❌ MISS'}")
print(f"  S F1 in [65,72]?{'✅ PASS' if 65<=per_class['S']['f1']<=72 else '❌ MISS'}")
print(f"  F F1 in [64,71]?{'✅ PASS' if 64<=per_class['F']['f1']<=71 else '❌ MISS'}")
print(f"  Acc >= 95.5%?   {'✅ PASS' if t_acc >= 95.5 else '❌ MISS'}")
