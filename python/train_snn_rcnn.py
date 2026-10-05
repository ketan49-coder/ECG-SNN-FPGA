"""
train_snn_rcnn.py — Spiking RCNN (Recurrent SNN)
=================================================
Adds recurrent connections to train_snn_v2.py baseline using snn.RLeaky.
Each hidden neuron now receives spikes from ALL neurons in the same layer
(via the V matrix), giving the layer temporal self-memory.

Why recurrent for ECG:
  - ECG patterns are temporal sequences (P → QRS → T wave)
  - Recurrent connections let the network "remember" what fired recently
  - This is the SNN equivalent of what LSTMs do for time-series
  - Unlike CNN convolutions, the V matrix is spike×weight = no multipliers on FPGA

Architecture: 128 → [16 + 16×16 recurrent] → [16 + 16×16 recurrent] → 5
Extra weights: 16×16 + 16×16 = 512 recurrent weights (fits in same BRAM tile)
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
# Hyperparameters (same as v2 baseline)
# ─────────────────────────────────────────────
INPUT_FEATURES   = 128
HIDDEN_1_NEURONS = 16
HIDDEN_2_NEURONS = 16
OUTPUT_CLASSES   = 5
NUM_STEPS        = 25
BETA             = 0.9375
THRESHOLD        = 0.5
BATCH_SIZE       = 128
NUM_EPOCHS       = 60
LEARNING_RATE    = 5e-3
DROPOUT_P        = 0.3
DATA_DIR         = "data"
MODEL_DIR        = "models"
CLASS_NAMES      = ['Normal (N)', 'Supraventricular (S)',
                    'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Q1.7 Quantization-Aware Training (same as v2)
# ─────────────────────────────────────────────
class Q1_7_Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        clamped   = torch.clamp(weights, -1.0, 0.9921875)
        quantized = torch.round(clamped * 128.0) / 128.0
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_q17(w):
    return Q1_7_Quantize.apply(w)


# ─────────────────────────────────────────────
# Focal Loss (same as v2)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets,
                               weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────
# Spiking RCNN Architecture
# Key change: snn.Leaky → snn.RLeaky
# RLeaky adds a learnable V matrix (16×16) that
# feeds each layer's own spike output back as input
# on the next time step — this is the recurrent connection
# ─────────────────────────────────────────────
class ECG_SNN_RCNN(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # ── Layer 1: 128 → 16 (with 16×16 recurrent V1 matrix) ──
        self.fc1   = nn.Linear(INPUT_FEATURES,    HIDDEN_1_NEURONS, bias=False)
        self.rlif1 = snn.RLeaky(
            beta             = BETA,
            threshold        = THRESHOLD,
            spike_grad       = spike_grad,
            reset_mechanism  = "zero",
            linear_features  = HIDDEN_1_NEURONS,  # size of V1 matrix (16×16)
            all_to_all       = True                # all neurons connect to all neurons
        )

        self.drop = nn.Dropout(p=dropout_p)

        # ── Layer 2: 16 → 16 (with 16×16 recurrent V2 matrix) ──
        self.fc2   = nn.Linear(HIDDEN_1_NEURONS, HIDDEN_2_NEURONS, bias=False)
        self.rlif2 = snn.RLeaky(
            beta             = BETA,
            threshold        = THRESHOLD,
            spike_grad       = spike_grad,
            reset_mechanism  = "zero",
            linear_features  = HIDDEN_2_NEURONS,  # size of V2 matrix (16×16)
            all_to_all       = True
        )

        # ── Output Layer: 16 → 5 (no recurrent, standard Leaky) ──
        # Output layer stays non-recurrent — we only need
        # temporal memory in hidden layers
        self.fc3   = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES, bias=False)
        self.lif3  = snn.Leaky(
            beta            = BETA,
            threshold       = THRESHOLD,
            spike_grad      = spike_grad,
            reset_mechanism = "zero"
        )

    def forward(self, x):
        # RLeaky needs BOTH membrane potential AND spike state initialised
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        mem3       = self.lif3.init_leaky()

        spk3_rec = []

        for _ in range(NUM_STEPS):
            # Layer 1: feedforward + recurrent
            # spk1 from previous timestep is fed back through V1
            cur1       = F.linear(x, quantize_q17(self.fc1.weight))
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            # rlif1 internally computes: cur_total = cur1 + V1 @ spk1_prev

            spk1_drop  = self.drop(spk1)

            # Layer 2: feedforward + recurrent
            cur2       = F.linear(spk1_drop, quantize_q17(self.fc2.weight))
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)
            # rlif2 internally computes: cur_total = cur2 + V2 @ spk2_prev

            # Output layer: standard (no recurrent)
            cur3       = F.linear(spk2, quantize_q17(self.fc3.weight))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec, dim=0)   # [T, B, 5]


# ─────────────────────────────────────────────
# Weight count helper
# ─────────────────────────────────────────────
def count_weights(net):
    total = sum(p.numel() for p in net.parameters())
    print(f"\nTotal trainable parameters: {total}")
    print(f"  FC1  (128×16):  {128*16:>6}  feedforward")
    print(f"  V1   (16×16):   {16*16:>6}  recurrent layer 1")
    print(f"  FC2  (16×16):   {16*16:>6}  feedforward")
    print(f"  V2   (16×16):   {16*16:>6}  recurrent layer 2")
    print(f"  FC3  (16×5):    {16*5:>6}  output")
    print(f"  Total:          {128*16 + 16*16 + 16*16 + 16*16 + 16*5:>6}  weights")
    bram_bytes = (128*16 + 16*16 + 16*16 + 16*16 + 16*5)
    print(f"  BRAM needed:    {bram_bytes} bytes "
          f"({bram_bytes/18432*100:.1f}% of one 18Kb tile)")


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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    return train_loader, test_loader, y_train


def compute_class_weights(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    total   = len(y_train)
    weights = total / (len(unique) * counts.astype(float))
    print("\nClass distribution + weights:")
    for c, cnt, w in zip(unique, counts, weights):
        print(f"  {CLASS_NAMES[c]:<30} count={cnt:6d}   weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


@torch.no_grad()
def per_class_accuracy(net, loader):
    net.eval()
    correct = torch.zeros(OUTPUT_CLASSES)
    total   = torch.zeros(OUTPUT_CLASSES)
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        preds = net(data).sum(dim=0).argmax(dim=1)
        for c in range(OUTPUT_CLASSES):
            mask        = (targets == c)
            total[c]   += mask.sum().item()
            correct[c] += (preds[mask] == c).sum().item()
    return 100.0 * correct / total.clamp(min=1)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_model():
    train_loader, test_loader, y_train = load_data()
    class_weights = compute_class_weights(y_train)

    net = ECG_SNN_RCNN().to(device)
    count_weights(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn   = FocalLoss(alpha=class_weights, gamma=2.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_test_acc = 0.0

    print(f"\nTraining ECG_SNN_RCNN | Steps={NUM_STEPS} | β={BETA} | "
          f"Threshold={THRESHOLD} | Dropout={DROPOUT_P}")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train%':>7} | "
          f"{'Test%':>7} | {'Gap':>6} | {'LR':>8}")
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        net.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            spk_out   = net(data)
            spike_sum = spk_out.sum(dim=0)
            loss      = loss_fn(spike_sum, targets)

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients — important for recurrent networks
            # (prevents exploding gradients through the V matrix)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds       = spike_sum.argmax(dim=1)
            correct    += (preds == targets).sum().item()
            total      += targets.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Evaluate on test set every epoch
        net.eval()
        t_correct = 0
        t_total   = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                preds      = net(data).sum(dim=0).argmax(dim=1)
                t_correct += (preds == targets).sum().item()
                t_total   += targets.size(0)

        test_acc = 100.0 * t_correct / t_total
        gap      = train_acc - test_acc

        gap_flag = ""
        if gap > 7:  gap_flag = " ← OVERFITTING"
        elif gap > 3: gap_flag = " ← watch"

        print(f"{epoch+1:>6} | {total_loss/len(train_loader):>8.4f} | "
              f"{train_acc:>6.2f}% | {test_acc:>6.2f}% | "
              f"{gap:>+5.1f}%{gap_flag} | {current_lr:.2e}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), f"{MODEL_DIR}/snn_rcnn_best.pth")
            print(f"         ★ New best saved → {best_test_acc:.2f}%")

        if (epoch + 1) % 10 == 0:
            accs = per_class_accuracy(net, test_loader)
            print(f"\n         Per-class accuracy (epoch {epoch+1}):")
            for c, name in enumerate(CLASS_NAMES):
                bar = "█" * int(accs[c] / 5)
                print(f"           {name:<30} {accs[c]:>6.2f}%  {bar}")
            print()

    print("=" * 80)
    print(f"Best Test Accuracy : {best_test_acc:.2f}%")
    print(f"Model saved to     : {MODEL_DIR}/snn_rcnn_best.pth")


if __name__ == "__main__":
    train_model()
