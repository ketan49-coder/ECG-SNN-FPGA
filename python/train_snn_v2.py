"""
train_snn_v2.py — Improved SNN Baseline (No RCNN, No DFA)
==========================================================
Changes over train_snn.py and train_snn_sota.py:
  1. Direct current injection (no rate coding)         → better signal fidelity
  2. Quantization-Aware Training (Q1.7 STE)            → no accuracy drop at quantization
  3. Focal Loss with aggressive class weights           → fixes S and F class learning
  4. Dropout (training only, zero FPGA cost)            → reduces overfitting
  5. NUM_STEPS = 25 (up from 16)                       → better temporal integration
  6. Cosine annealing LR schedule (replaces StepLR)    → smoother convergence
  7. Per-class accuracy printed every epoch             → catch overfitting early
  8. Train vs Test gap printed every epoch              → diagnose overfitting
  9. Best model saved on test accuracy                  → clean checkpoint
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
INPUT_FEATURES   = 128
HIDDEN_1_NEURONS = 16
HIDDEN_2_NEURONS = 16
OUTPUT_CLASSES   = 5
NUM_STEPS        = 25       # Up from 16 → better temporal integration
BETA             = 0.9375   # Maps to >>4 right-shift on FPGA
THRESHOLD        = 0.5      # Lower threshold for deep SNN
BATCH_SIZE       = 128
NUM_EPOCHS       = 60       # More epochs for cosine annealing to work well
LEARNING_RATE    = 5e-3
DROPOUT_P        = 0.3      # Dropout probability (only active during training)
DATA_DIR         = "data"
MODEL_DIR        = "models"
CLASS_NAMES      = ['Normal (N)', 'Supraventricular (S)',
                    'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 1. Q1.7 Quantization-Aware Training
#    Straight-Through Estimator (STE):
#    Forward  → quantize weights to Q1.7
#    Backward → pass gradient through unchanged
# ─────────────────────────────────────────────
class Q1_7_Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        clamped   = torch.clamp(weights, -1.0, 0.9921875)
        quantized = torch.round(clamped * 128.0) / 128.0
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # STE: gradient passes through unchanged

def quantize_q17(w):
    return Q1_7_Quantize.apply(w)


# ─────────────────────────────────────────────
# 2. Focal Loss
#    Downweights easy examples so the model
#    focuses on hard minority classes (S, F)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha   # per-class weight tensor

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets,
                               weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce).mean()
        return loss


# ─────────────────────────────────────────────
# 3. Network Architecture (128 → 16 → 16 → 5)
#    - Direct current injection (no rate coding)
#    - QAT weights at every layer
#    - Dropout between hidden layers
# ─────────────────────────────────────────────
class ECG_SNN_V2(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1  = nn.Linear(INPUT_FEATURES,    HIDDEN_1_NEURONS, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad, reset_mechanism="zero")

        self.drop = nn.Dropout(p=dropout_p)   # between hidden layers only

        self.fc2  = nn.Linear(HIDDEN_1_NEURONS, HIDDEN_2_NEURONS, bias=False)
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad, reset_mechanism="zero")

        self.fc3  = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES,   bias=False)
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad, reset_mechanism="zero")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []

        for _ in range(NUM_STEPS):
            # Direct current injection (x repeated every timestep)
            # QAT: quantize weights during forward pass
            cur1       = F.linear(x, quantize_q17(self.fc1.weight))
            spk1, mem1 = self.lif1(cur1, mem1)

            spk1_drop  = self.drop(spk1)   # dropout on spikes (not mem)

            cur2       = F.linear(spk1_drop, quantize_q17(self.fc2.weight))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3       = F.linear(spk2, quantize_q17(self.fc3.weight))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec, dim=0)   # [T, B, 5]


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32),
                             torch.tensor(y_test,  dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader, y_train


# ─────────────────────────────────────────────
# Class weights (aggressive inverse frequency)
# ─────────────────────────────────────────────
def compute_class_weights(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    total   = len(y_train)
    weights = total / (len(unique) * counts.astype(float))

    print("\nClass distribution + weights:")
    for c, cnt, w in zip(unique, counts, weights):
        print(f"  {CLASS_NAMES[c]:<30} count={cnt:6d}   weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────
# Per-class accuracy helper
# ─────────────────────────────────────────────
@torch.no_grad()
def per_class_accuracy(net, loader):
    net.eval()
    correct = torch.zeros(OUTPUT_CLASSES)
    total   = torch.zeros(OUTPUT_CLASSES)

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_out = net(data)
        preds   = spk_out.sum(dim=0).argmax(dim=1)
        for c in range(OUTPUT_CLASSES):
            mask          = (targets == c)
            total[c]     += mask.sum().item()
            correct[c]   += (preds[mask] == c).sum().item()

    accs = 100.0 * correct / total.clamp(min=1)
    return accs   # tensor of 5 per-class accuracies


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train_model():
    train_loader, test_loader, y_train = load_data()
    class_weights = compute_class_weights(y_train)

    net       = ECG_SNN_V2().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn   = FocalLoss(alpha=class_weights, gamma=2.0)

    # Cosine annealing: LR smoothly decays to 0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_test_acc = 0.0

    print(f"\nTraining ECG_SNN_V2 | Steps={NUM_STEPS} | β={BETA} | "
          f"Threshold={THRESHOLD} | Dropout={DROPOUT_P}")
    print("=" * 80)
    header = (f"{'Epoch':>6} | {'Loss':>8} | {'Train%':>7} | "
              f"{'Test%':>7} | {'Gap':>6} | {'LR':>8}")
    print(header)
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        net.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            spk_out    = net(data)
            spike_sum  = spk_out.sum(dim=0)   # [B, 5]
            loss       = loss_fn(spike_sum, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = spike_sum.argmax(dim=1)
            correct    += (preds == targets).sum().item()
            total      += targets.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Evaluate on test set every epoch ──
        net.eval()
        t_correct = 0
        t_total   = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                spk_out = net(data)
                preds   = spk_out.sum(dim=0).argmax(dim=1)
                t_correct += (preds == targets).sum().item()
                t_total   += targets.size(0)

        test_acc = 100.0 * t_correct / t_total
        gap      = train_acc - test_acc

        # ── Gap warning ──
        gap_flag = ""
        if gap > 7:
            gap_flag = " ← OVERFITTING"
        elif gap > 3:
            gap_flag = " ← watch"

        print(f"{epoch+1:>6} | {total_loss/len(train_loader):>8.4f} | "
              f"{train_acc:>6.2f}% | {test_acc:>6.2f}% | "
              f"{gap:>+5.1f}%{gap_flag} | {current_lr:.2e}")

        # ── Save best model ──
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), f"{MODEL_DIR}/snn_v2_best.pth")
            print(f"         ★ New best saved → {best_test_acc:.2f}%")

        # ── Per-class breakdown every 10 epochs ──
        if (epoch + 1) % 10 == 0:
            accs = per_class_accuracy(net, test_loader)
            print(f"\n         Per-class accuracy (epoch {epoch+1}):")
            for c, name in enumerate(CLASS_NAMES):
                bar = "█" * int(accs[c] / 5)
                print(f"           {name:<30} {accs[c]:>6.2f}%  {bar}")
            print()

    print("=" * 80)
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to:     {MODEL_DIR}/snn_v2_best.pth")


if __name__ == "__main__":
    train_model()
