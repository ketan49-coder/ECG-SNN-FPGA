import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Training SNN on 131-Feature Dataset (Robust Normalization + 3 RR)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FEATURES, HIDDEN_1, HIDDEN_2, OUTPUT_CLASSES = 131, 32, 32, 4
NUM_STEPS, BETA, THRESHOLD = 25, 0.9375, 0.5
BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE = 128, 60, 2e-3
DATA_DIR = "data_ds_step3"
MODEL_DIR = "/content/drive/MyDrive/SNN_Models" if os.path.exists("/content/drive/MyDrive/SNN_Models") else "models"
CLASS_NAMES = ['Normal (N)', 'Supra (S)', 'Ventricular (V)', 'Fusion (F)']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_train.npy"), dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_train.npy"), dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_val.npy"),   dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_val.npy"),   dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.tensor(np.load(f"{DATA_DIR}/X_test.npy"),  dtype=torch.float32), torch.tensor(np.load(f"{DATA_DIR}/y_test.npy"),  dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=False)

    net = ECG_SuperSpike().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-5)
    loss_fn = VanRossumLoss()
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_f1, best_ep = 0.0, 0
    ckpt_path = f"{MODEL_DIR}/snn_superspike_ds1_ds2_step3_best.pth"

    print("\n" + "="*80)
    print("STEP 3 RUN: ROBUST NORMALIZATION + 3 RR TIMING FEATURES (131 INPUTS)")
    print("="*80)
    print("Testing Pre-Registered Prediction 1: Record 232 S recall rises from 2.4% to > 30.0%")
    print("Testing Pre-Registered Prediction 2: Overall DS2 S recall rises from 13.06% to > 40.0%")
    print("Testing Pre-Registered Prediction 3: False S across flooded records falls by >= 50%")
    print("Testing Pre-Registered Prediction 4: DS2 Test Macro-F1 rises from 45.90% to > 55.0%")
    print("-" * 80)

    for ep in range(NUM_EPOCHS):
        net.train()
        s_loss, cor, tot = 0.0, 0, 0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            out = net(d); loss = loss_fn(out, t)
            opt.zero_grad(); loss.backward(); opt.step()
            s_loss += loss.item()
            cor += (out.sum(0).argmax(1) == t).sum().item(); tot += t.size(0)
        sched.step()
        val_acc, val_f1, _, _ = evaluate(net, val_loader)
        
        star = ""
        if val_f1 > best_val_f1:
            best_val_f1, best_ep = val_f1, ep + 1
            torch.save(net.state_dict(), ckpt_path)
            star = " ★ BEST F1"

        print(f"Ep {ep+1:2d} | Loss: {s_loss/len(train_loader):.4f} | Train: {100.0*cor/tot:.2f}% | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%{star}")

    print(f"\nTraining Complete! Selected Checkpoint from Epoch {best_ep} (Val Macro-F1: {best_val_f1:.2f}%)")
    print("\n" + "="*80 + "\nFINAL INDEPENDENT EVALUATION ON UNSEEN DS2 PATIENTS (STEP 3):\n" + "="*80)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc, test_f1, targets, preds = evaluate(net, test_loader)
    print(f"DS2 Test Accuracy: {test_acc:.2f}% | DS2 Macro-F1: {test_f1:.2f}%\n")
    print(classification_report(targets, preds, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(targets, preds))

if __name__ == "__main__":
    main()
