import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score

INPUT_FEATURES, HIDDEN_1, HIDDEN_2, OUTPUT_CLASSES = 131, 32, 32, 4
NUM_STEPS, BETA, THRESHOLD = 25, 0.9375, 0.5
BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = 128, 60, 2e-3
DATA_DIR = "data_ds1_svdb"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training DS1 + SVDB Model on device: {device} (Seed {SEED})")

class VanRossumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.decay = 0.9375
    def forward(self, spk, tgt):
        T, B, C = spk.shape
        t_spk = torch.zeros(T, B, C, device=spk.device)
        for b in range(B): t_spk[:, b, tgt[b]] = 1.0
        f_out, f_tgt = torch.zeros_like(spk), torch.zeros_like(t_spk)
        tr_o, tr_t = torch.zeros(B, C, device=spk.device), torch.zeros(B, C, device=spk.device)
        for t in range(T):
            tr_o = tr_o * self.decay + spk[t]; tr_t = tr_t * self.decay + t_spk[t]
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
            s1, m1 = self.lif1(self.fc1(x), m1); s1 = self.drop(s1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            spk_rec.append(torch.zeros(x.size(0), OUTPUT_CLASSES, device=x.device) if not d_mask[t] else s3)
        return torch.stack(spk_rec)

def evaluate(net, loader):
    net.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for d, t in loader:
            all_p.extend(net(d.to(device)).sum(0).argmax(1).cpu().numpy())
            all_t.extend(t.numpy())
    all_p, all_t = np.array(all_p), np.array(all_t)
    acc = 100.0 * np.mean(all_p == all_t)
    f1  = 100.0 * f1_score(all_t, all_p, average='macro', zero_division=0)
    return acc, f1, all_t, all_p

print("Loading datasets...")
train_loader = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_train.npy"), dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_train.npy"), dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_val.npy"),   dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_val.npy"),   dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_test.npy"),  dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_test.npy"),  dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)

net = ECG_SuperSpike().to(device)
opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-5)
loss_fn = VanRossumLoss()

best_val_f1, best_ep = 0.0, 0
ckpt_path = f"{MODEL_DIR}/snn_ds1_svdb_seed42_best.pth"

print(f"\n---> Training DS1 + SVDB on Seed {SEED} (60 epochs, Cosine Annealing, Peak Val F1)...")
for ep in range(1, NUM_EPOCHS + 1):
    net.train()
    tot_loss, nb = 0.0, 0
    for d, t in train_loader:
        d, t = d.to(device), t.to(device)
        out = net(d); loss = loss_fn(out, t)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item(); nb += 1
    sched.step()
    
    val_acc, val_f1, _, _ = evaluate(net, val_loader)
    is_best = val_f1 > best_val_f1
    if is_best:
        best_val_f1, best_ep = val_f1, ep
        torch.save(net.state_dict(), ckpt_path)
        
    if is_best or ep % 5 == 0 or ep == NUM_EPOCHS:
        star = " (*BEST VAL F1*)" if is_best else ""
        print(f"  [Ep {ep:02d}/60] Loss: {tot_loss/nb:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%{star}", flush=True)

print(f"\nLoading Best Checkpoint from Epoch {best_ep} (Val Macro-F1: {best_val_f1:.2f}%)...")
net.load_state_dict(torch.load(ckpt_path, map_location=device))

test_acc, test_f1, targets, preds = evaluate(net, test_loader)
cm = confusion_matrix(targets, preds, labels=[0, 1, 2, 3])

print("\n" + "=" * 80)
print(f"DS1 + SVDB FINAL EVALUATION ON FROZEN DS2 TEST SET (Seed {SEED})")
print("=" * 80)
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Macro-F1: {test_f1:.2f}%\n")
print("Classification Report:")
print(classification_report(targets, preds, target_names=['Normal (N)', 'Supra (S)', 'Ventricular (V)', 'Fusion (F)'], digits=4))

print("Confusion Matrix:")
print("         Pred N   Pred S   Pred V   Pred F")
for idx, cname in enumerate(['True N', 'True S', 'True V', 'True F']):
    print(f"{cname:6s}: {cm[idx, 0]:7d}  {cm[idx, 1]:7d}  {cm[idx, 2]:7d}  {cm[idx, 3]:7d}")

rec_S = 100.0 * cm[1, 1] / max(1, cm[1].sum())
prec_S = 100.0 * cm[1, 1] / max(1, cm[:, 1].sum())
print(f"\nPooled S Recall: {rec_S:.2f}% | S Precision: {prec_S:.2f}%")
print("=" * 80)
