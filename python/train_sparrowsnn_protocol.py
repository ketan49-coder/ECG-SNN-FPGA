"""
SparrowSNN-Protocol 4-Class Random Split Benchmark
===================================================
Evaluates multiplier-free Spartan-7 SNN (128 inputs -> 32 -> 32 -> 4 outputs)
under the exact protocol of SparrowSNN:
  - 44 non-paced records of MIT-BIH (excluding 102, 104, 107, 217).
  - 4 AAMI classes (N, S, V, F).
  - 60% Train / 20% Val / 20% Test random beat-level stratified split.
  - SMOTE applied exclusively to the training portion.
  - Plain Van Rossum distance loss (tau = 0.9375), 16 timesteps, Cosine Annealing.
  - Evaluated across 3 seeds: [42, 123, 999].
"""

import os
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    os.system("pip install -q imbalanced-learn")
    from imblearn.over_sampling import SMOTE

import wfdb

# 1. Hyperparameters & Architecture
INPUT_FEATURES = 128
HIDDEN_1 = 32
HIDDEN_2 = 32
OUTPUT_CLASSES = 4
NUM_STEPS = 16          # 16 timesteps matching Spartan-7 hardware
BETA = 0.9375           # Bit-shift leak: 1 - 1/16 (V >> 4)
THRESHOLD = 0.5
BATCH_SIZE = 256
NUM_EPOCHS = 60
LEARNING_RATE = 2e-3
SEEDS = [42, 123, 999]

EXCLUDED_PACED = {'102', '104', '107', '217'}
AAMI_4_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running SparrowSNN Protocol on device: {device}")

# 2. Locate MIT-BIH Records
mit_candidates = [r for r, _, f in os.walk('/content', followlinks=True) if '100.dat' in f]
MIT_DIR = mit_candidates[0] if mit_candidates else 'mitdb'
print(f"Using MIT-BIH dataset directory: {MIT_DIR}")

# 3. Extract 44 Non-Paced Records (128-Sample Window)
cache_path = "mitbih_sparrow_4class_128.npz"
if os.path.exists(cache_path):
    print(f"Loading beats from local cache: {cache_path}...")
    cached = np.load(cache_path)
    X_all, y_all = cached['X'], cached['y']
else:
    all_recs = sorted(list(set([
        f.split('.')[0] for f in os.listdir(MIT_DIR)
        if f.endswith('.dat') and f.split('.')[0] not in EXCLUDED_PACED
    ])))
    print(f"Extracting {len(all_recs)} non-paced records...")

    X_list, y_list = [], []
    for rec in all_recs:
        p_rec = os.path.join(MIT_DIR, rec)
        try:
            r = wfdb.rdrecord(p_rec, channels=[0])
            sig = r.p_signal.flatten()
            ann = wfdb.rdann(p_rec, 'atr')

            med = np.median(sig)
            mad = np.median(np.abs(sig - med)) * 1.4826 + 1e-4
            sig_norm = np.clip(((sig - med) / mad + 4.0) / 14.0, 0.0, 1.0)

            for i in range(len(ann.sample)):
                p, s = ann.sample[i], ann.symbol[i]
                if s not in AAMI_4_MAP or (p - 64 < 0) or (p + 64 > len(sig_norm)):
                    continue
                beat = sig_norm[p - 64 : p + 64]
                if len(beat) == 128:
                    X_list.append(beat)
                    y_list.append(AAMI_4_MAP[s])
        except Exception as e:
            print(f"  Warning on {rec}: {e}")

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.int64)
    np.savez_compressed(cache_path, X=X_all, y=y_all)
    print(f"Extracted and cached {len(y_all)} beats.")

print(f"\nTotal Dataset: {len(y_all)} beats.")
counts = np.bincount(y_all, minlength=4)
print(f"Class Breakdown: Normal={counts[0]}, S={counts[1]}, V={counts[2]}, F={counts[3]}")

# 4. Model Architecture & Plain Van Rossum Loss
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
        tr_o, tr_t = torch.zeros(B, C, device=spk.device), torch.zeros(B, C, device=spk.device)
        for t in range(T):
            tr_o = tr_o * self.decay + spk[t]
            tr_t = tr_t * self.decay + t_spk[t]
            f_out[t], f_tgt[t] = tr_o, tr_t
        return F.mse_loss(f_out, f_tgt, reduction='sum') / B

class ECG_Spartan7_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_1, bias=False)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2, bias=False)
        self.fc3 = nn.Linear(HIDDEN_2, OUTPUT_CLASSES, bias=False)
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
            d = d.to(device)
            out = net(d)
            pred = out.sum(dim=0).argmax(dim=1).cpu().numpy()
            all_p.extend(pred)
            all_t.extend(t.numpy())
    all_p, all_t = np.array(all_p), np.array(all_t)
    acc = 100.0 * np.mean(all_p == all_t)
    f1  = 100.0 * f1_score(all_t, all_p, average='macro', zero_division=0)
    return acc, f1, all_t, all_p

# 5. Multi-Seed Loop (3 Seeds)
results = []
for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"---> STARTING SEED {seed} (60/20/20 Random Split + SMOTE on Train Only)")
    print("=" * 80)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 60% Train, 20% Val, 20% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=seed, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
    )

    print(f"Partition Sizes: Train={len(y_train)} (60%), Val={len(y_val)} (20%), Test={len(y_test)} (20%)")
    print("Applying SMOTE exclusively to the training portion...")
    smote = SMOTE(random_state=seed)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    sm_counts = np.bincount(y_train_sm, minlength=4)
    print(f"SMOTE Balanced Train: N={sm_counts[0]}, S={sm_counts[1]}, V={sm_counts[2]}, F={sm_counts[3]}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_sm, dtype=torch.float32), torch.tensor(y_train_sm, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    net = ECG_Spartan7_SNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-5)
    loss_fn = VanRossumLoss(tau_decay=BETA).to(device)

    best_val_f1, best_ep = 0.0, 0
    best_state = None

    for ep in range(1, NUM_EPOCHS + 1):
        net.train()
        tot_loss, nb = 0.0, 0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            out = net(d)
            loss = loss_fn(out, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            nb += 1
        sched.step()

        val_acc, val_f1, _, _ = evaluate(net, val_loader)
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1, best_ep = val_f1, ep
            best_state = net.state_dict().copy()

        if is_best or ep % 10 == 0 or ep == NUM_EPOCHS:
            star = " (*BEST VAL F1*)" if is_best else ""
            print(f"  [Ep {ep:02d}/60] Loss: {tot_loss/nb:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%{star}", flush=True)

    print(f"\nEvaluating Best Checkpoint (Epoch {best_ep}, Val F1 {best_val_f1:.2f}%) on Un-Augmented Test Set...")
    net.load_state_dict(best_state)
    t_acc, t_f1, t_true, t_pred = evaluate(net, test_loader)
    cm = confusion_matrix(t_true, t_pred, labels=[0, 1, 2, 3])

    rec_N = 100.0 * cm[0, 0] / max(1, cm[0].sum())
    rec_S = 100.0 * cm[1, 1] / max(1, cm[1].sum())
    rec_V = 100.0 * cm[2, 2] / max(1, cm[2].sum())
    rec_F = 100.0 * cm[3, 3] / max(1, cm[3].sum())

    print(f"\nSeed {seed} Test Results: Acc: {t_acc:.2f}% | Macro-F1: {t_f1:.2f}%")
    print(f"Per-Class Recall: N: {rec_N:.2f}% | S: {rec_S:.2f}% | V: {rec_V:.2f}% | F: {rec_F:.2f}%")
    print("Confusion Matrix:")
    print("         Pred N   Pred S   Pred V   Pred F")
    for idx, cname in enumerate(['True N', 'True S', 'True V', 'True F']):
        print(f"{cname:6s}: {cm[idx, 0]:7d}  {cm[idx, 1]:7d}  {cm[idx, 2]:7d}  {cm[idx, 3]:7d}")

    results.append({
        'seed': seed, 'acc': t_acc, 'f1': t_f1,
        'rec_N': rec_N, 'rec_S': rec_S, 'rec_V': rec_V, 'rec_F': rec_F,
        'cm': cm
    })

# 6. Aggregate Summary
accs = [r['acc'] for r in results]
f1s = [r['f1'] for r in results]
r_Ns = [r['rec_N'] for r in results]
r_Ss = [r['rec_S'] for r in results]
r_Vs = [r['rec_V'] for r in results]
r_Fs = [r['rec_F'] for r in results]

print("\n" + "=" * 80)
print("FINAL BENCHMARK: Random Beat Split, Protocol of SparrowSNN (3 Seeds)")
print("=" * 80)
print(f"Overall Accuracy:  {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
print(f"Macro-F1 Score:    {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%\n")
print("Per-Class Recall:")
print(f"  Normal (N) Recall:           {np.mean(r_Ns):.2f}% ± {np.std(r_Ns):.2f}%")
print(f"  Supraventricular (S) Recall: {np.mean(r_Ss):.2f}% ± {np.std(r_Ss):.2f}%")
print(f"  Ventricular (V) Recall:      {np.mean(r_Vs):.2f}% ± {np.std(r_Vs):.2f}%")
print(f"  Fusion (F) Recall:           {np.mean(r_Fs):.2f}% ± {np.std(r_Fs):.2f}%")
print("=" * 80)
