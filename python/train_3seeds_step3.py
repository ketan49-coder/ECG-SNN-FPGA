import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# 3-Seed Evaluation Script on Winning Configuration (Step 3: 131 Inputs)
# Evaluates Seeds 42, 123, 999 under both Raw and Decision-Offset (+6 S, -15 F)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FEATURES, HIDDEN_1, HIDDEN_2, OUTPUT_CLASSES = 131, 32, 32, 4
NUM_STEPS, BETA, THRESHOLD = 25, 0.9375, 0.5
BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = 128, 60, 2e-3
DATA_DIR = "data_ds_step3"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEEDS = [42, 123, 999]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing 3-seed benchmark on device: {device}")

class VanRossumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.decay = 0.9375
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
        return F.mse_loss(f_out, f_tgt, reduction='none').sum(dim=(0, 2)).mean()

class ECG_SuperSpike(nn.Module):
    def __init__(self):
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_1, bias=False)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2, bias=False)
        self.fc3 = nn.Linear(HIDDEN_2, OUTPUT_CLASSES, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism="zero")
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism="zero")
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        spk_rec = []
        d_mask = torch.rand(NUM_STEPS) > 0.1 if self.training else torch.ones(NUM_STEPS, dtype=torch.bool)
        for t in range(NUM_STEPS):
            s1, m1 = self.lif1(self.fc1(x), m1)
            s1 = self.drop(s1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            spk_rec.append(torch.zeros(x.size(0), OUTPUT_CLASSES, device=x.device) if not d_mask[t] else s3)
        return torch.stack(spk_rec)

def evaluate(net, loader, s_bonus=0, f_penalty=0):
    net.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for d, t in loader:
            counts = net(d.to(device)).sum(0)
            if s_bonus != 0:
                counts[:, 1] += s_bonus
            if f_penalty != 0:
                counts[:, 3] -= f_penalty
            all_p.extend(counts.argmax(1).cpu().numpy())
            all_t.extend(t.numpy())
    all_p, all_t = np.array(all_p), np.array(all_t)
    acc = 100.0 * np.mean(all_p == all_t)
    f1  = 100.0 * f1_score(all_t, all_p, average='macro', zero_division=0)
    cm  = confusion_matrix(all_t, all_p, labels=[0, 1, 2, 3])
    
    rec_N = 100.0 * cm[0, 0] / max(1, cm[0].sum())
    rec_S = 100.0 * cm[1, 1] / max(1, cm[1].sum())
    rec_V = 100.0 * cm[2, 2] / max(1, cm[2].sum())
    rec_F = 100.0 * cm[3, 3] / max(1, cm[3].sum())
    
    prec_N = 100.0 * cm[0, 0] / max(1, cm[:, 0].sum())
    prec_S = 100.0 * cm[1, 1] / max(1, cm[:, 1].sum())
    prec_V = 100.0 * cm[2, 2] / max(1, cm[:, 2].sum())
    prec_F = 100.0 * cm[3, 3] / max(1, cm[:, 3].sum())
    
    return {
        'acc': acc, 'f1': f1,
        'rec_N': rec_N, 'rec_S': rec_S, 'rec_V': rec_V, 'rec_F': rec_F,
        'prec_N': prec_N, 'prec_S': prec_S, 'prec_V': prec_V, 'prec_F': prec_F,
        'cm': cm
    }

def run_seed(seed, train_loader, val_loader, test_loader):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    net = ECG_SuperSpike().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-5)
    loss_fn = VanRossumLoss()
    
    best_val_f1, best_ep = 0.0, 0
    ckpt_path = f"{MODEL_DIR}/snn_step3_seed_{seed}_best.pth"
    
    print(f"\n================================================================================")
    print(f"TRAINING SEED {seed} (60 Epochs, Cosine Annealing, Checkpoint on Peak Val Macro-F1)")
    print(f"================================================================================")
    
    for ep in range(1, NUM_EPOCHS + 1):
        net.train()
        total_loss, n_batches = 0.0, 0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            out = net(d)
            loss = loss_fn(out, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        
        val_res = evaluate(net, val_loader)
        is_best = val_res['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_res['f1']
            best_ep = ep
            torch.save(net.state_dict(), ckpt_path)
            
        if is_best or ep % 5 == 0 or ep == NUM_EPOCHS:
            best_tag = " (*BEST VAL F1*)" if is_best else ""
            print(f"  [Seed {seed} | Ep {ep:02d}/60] Loss: {total_loss/n_batches:.4f} | Val Acc: {val_res['acc']:.2f}% | Val Macro-F1: {val_res['f1']:.2f}%{best_tag}", flush=True)
            
    # Load winning checkpoint
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # 1. Raw Evaluation
    res_raw = evaluate(net, test_loader, s_bonus=0, f_penalty=0)
    # 2. Offset Evaluation (+6 S, -15 F)
    res_off = evaluate(net, test_loader, s_bonus=6, f_penalty=15)
    
    print(f"\n--> Seed {seed} Final Evaluation on DS2 (Selected at Epoch {best_ep}, Val F1 {best_val_f1:.2f}%):")
    print(f"    Raw:     Acc: {res_raw['acc']:.2f}% | Macro-F1: {res_raw['f1']:.2f}% | S Rec: {res_raw['rec_S']:.2f}% (Prec {res_raw['prec_S']:.2f}%) | V Rec: {res_raw['rec_V']:.2f}% | F Rec: {res_raw['rec_F']:.2f}%")
    print(f"    Offsets: Acc: {res_off['acc']:.2f}% | Macro-F1: {res_off['f1']:.2f}% | S Rec: {res_off['rec_S']:.2f}% (Prec {res_off['prec_S']:.2f}%) | V Rec: {res_off['rec_V']:.2f}% | F Rec: {res_off['rec_F']:.2f}%")
    
    return {
        'seed': seed, 'best_ep': best_ep, 'val_f1': best_val_f1,
        'raw': res_raw, 'off': res_off
    }

def print_summary_table(title, results, key):
    print("\n" + "=" * 92)
    print(f"{title} (MEAN ± STD)")
    print("=" * 92)
    print(f"| {'Metric':<22} | {'Seed 42':<10} | {'Seed 123':<10} | {'Seed 999':<10} | {'Mean ± Std':<16} |")
    print("|:-----------------------|:-----------|:-----------|:-----------|:-----------------|")
    
    metrics = [
        ('Overall Accuracy (%)', 'acc'),
        ('Macro-F1 Score (%)',   'f1'),
        ('S Recall (%)',         'rec_S'),
        ('S Precision (%)',      'prec_S'),
        ('V Recall (%)',         'rec_V'),
        ('V Precision (%)',      'prec_V'),
        ('F Recall (%)',         'rec_F'),
        ('F Precision (%)',      'prec_F'),
        ('N Recall (%)',         'rec_N'),
        ('N Precision (%)',      'prec_N'),
    ]
    
    for label, m_key in metrics:
        v = [r[key][m_key] for r in results]
        mean, std = np.mean(v), np.std(v)
        print(f"| {label:<22} | {v[0]:>9.2f}% | {v[1]:>9.2f}% | {v[2]:>9.2f}% | **{mean:5.2f} ± {std:4.2f}%** |")
    print("=" * 92)

def main():
    print("Loading prepared DS1 and DS2 datasets...")
    train_loader = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_train.npy"), dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_train.npy"), dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_val.npy"),   dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_val.npy"),   dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_test.npy"),  dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_test.npy"),  dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
    
    results = []
    for s in SEEDS:
        res = run_seed(s, train_loader, val_loader, test_loader)
        results.append(res)
        
    print_summary_table("STEP 3 RAW (3-SEED ROBUSTNESS)", results, 'raw')
    print_summary_table("STEP 3 + DECISION OFFSETS (+6 S, -15 F)", results, 'off')

if __name__ == "__main__":
    main()
