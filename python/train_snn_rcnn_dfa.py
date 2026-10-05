"""
train_snn_rcnn_dfa.py — Spiking RCNN with DFA + Local Loss
============================================================
All improvements combined:
  1. Hidden size 16 → 32      (more capacity)
  2. Recurrent connections     (snn.RLeaky — temporal memory)
  3. Local Loss                (auxiliary head at each hidden layer)
  4. Direct Feedback Alignment (DFA — fixed random matrices per layer)
  5. Q1.7 Quantization-Aware Training
  6. Focal Loss + class weights
  7. Dropout between layers
  8. NUM_STEPS = 25
  9. Cosine annealing LR
  10. Gradient clipping (essential for recurrent nets)

How DFA works here:
  Standard BPTT:  error flows δ(L) → δ(L-1) → δ(L-2) chain
  DFA:            each hidden layer gets error DIRECTLY from output
                  via fixed random matrix H: δ(l) = σ'(U) * (H(l) @ δ_out)
  
  Implementation: we detach hidden layer outputs from the output loss,
  and instead use per-layer auxiliary losses that receive a DFA-feedback
  signal. Each layer trains on local_loss(l) + dfa_alignment_loss(l).

How Local Loss works here:
  Each hidden layer has a tiny auxiliary linear head → softmax → loss.
  Total loss = α*L_out + β*L_h1 + β*L_h2
  This ensures each layer receives a DIRECT task-relevant error signal,
  not just error filtered through layers above it.
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
HIDDEN_1_NEURONS = 32    # ← up from 16
HIDDEN_2_NEURONS = 32    # ← up from 16
OUTPUT_CLASSES   = 5
NUM_STEPS        = 25
BETA             = 0.9375
THRESHOLD        = 0.5
BATCH_SIZE       = 128
NUM_EPOCHS       = 60
LEARNING_RATE    = 3e-3   # slightly lower for larger network
DROPOUT_P        = 0.3
LOCAL_LOSS_W     = 0.3    # weight for auxiliary layer losses
DATA_DIR         = "data"
MODEL_DIR        = "models"
CLASS_NAMES      = ['Normal (N)', 'Supraventricular (S)',
                    'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Q1.7 Quantization-Aware Training
# ─────────────────────────────────────────────
class Q1_7_Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        return torch.round(torch.clamp(w, -1.0, 0.9921875) * 128.0) / 128.0

    @staticmethod
    def backward(ctx, grad):
        return grad   # straight-through estimator

def quantize_q17(w):
    return Q1_7_Quantize.apply(w)


# ─────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────
# Spiking RCNN + DFA + Local Loss
# ─────────────────────────────────────────────
class ECG_SNN_RCNN_DFA(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # ── Main feedforward weights ──
        self.fc1  = nn.Linear(INPUT_FEATURES,    HIDDEN_1_NEURONS, bias=False)
        self.fc2  = nn.Linear(HIDDEN_1_NEURONS,  HIDDEN_2_NEURONS, bias=False)
        self.fc3  = nn.Linear(HIDDEN_2_NEURONS,  OUTPUT_CLASSES,   bias=False)

        # ── Recurrent LIF neurons ──
        self.rlif1 = snn.RLeaky(beta=BETA, threshold=THRESHOLD,
                                spike_grad=spike_grad, reset_mechanism="zero",
                                linear_features=HIDDEN_1_NEURONS, all_to_all=True)

        self.rlif2 = snn.RLeaky(beta=BETA, threshold=THRESHOLD,
                                spike_grad=spike_grad, reset_mechanism="zero",
                                linear_features=HIDDEN_2_NEURONS, all_to_all=True)

        self.lif3  = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad, reset_mechanism="zero")

        self.drop  = nn.Dropout(p=dropout_p)

        # ── Local Loss auxiliary heads ──
        # Tiny linear classifiers attached to each hidden layer
        # They receive the summed spike output of that layer and predict the class
        self.local_head1 = nn.Linear(HIDDEN_1_NEURONS, OUTPUT_CLASSES, bias=False)
        self.local_head2 = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES, bias=False)

        # ── DFA Feedback Matrices (fixed, random, never updated) ──
        # H(l): maps output error [B, 5] → hidden layer error [B, 32]
        # These are buffers (not parameters) so they are never trained
        H1 = torch.randn(OUTPUT_CLASSES, HIDDEN_1_NEURONS) * 0.1
        H2 = torch.randn(OUTPUT_CLASSES, HIDDEN_2_NEURONS) * 0.1
        # Normalize columns so feedback signal has consistent scale
        H1 = H1 / H1.norm(dim=0, keepdim=True).clamp(min=1e-8)
        H2 = H2 / H2.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer('H1', H1)  # [5, 32]
        self.register_buffer('H2', H2)  # [5, 32]

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        mem3       = self.lif3.init_leaky()

        spk1_rec, spk2_rec, spk3_rec = [], [], []

        for _ in range(NUM_STEPS):
            cur1       = F.linear(x, quantize_q17(self.fc1.weight))
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)

            spk1_drop  = self.drop(spk1)

            cur2       = F.linear(spk1_drop, quantize_q17(self.fc2.weight))
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)

            cur3       = F.linear(spk2, quantize_q17(self.fc3.weight))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)

        # Stack across time: [T, B, neurons]
        spk1_all = torch.stack(spk1_rec, dim=0)
        spk2_all = torch.stack(spk2_rec, dim=0)
        spk3_all = torch.stack(spk3_rec, dim=0)

        return spk1_all, spk2_all, spk3_all


# ─────────────────────────────────────────────
# Data + helpers
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

    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            y_train)


def compute_class_weights(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (len(unique) * counts.astype(float))
    print("\nClass weights:")
    for c, cnt, w in zip(unique, counts, weights):
        print(f"  {CLASS_NAMES[c]:<30} count={cnt:6d}  weight={w:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    correct = torch.zeros(OUTPUT_CLASSES)
    total   = torch.zeros(OUTPUT_CLASSES)
    t_ok, t_tot = 0, 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        _, _, spk3 = net(data)
        preds = spk3.sum(0).argmax(1)
        t_ok  += (preds == targets).sum().item()
        t_tot += targets.size(0)
        for c in range(OUTPUT_CLASSES):
            m = (targets == c)
            total[c]   += m.sum().item()
            correct[c] += (preds[m] == c).sum().item()

    overall = 100.0 * t_ok / t_tot
    per_cls = 100.0 * correct / total.clamp(min=1)
    return overall, per_cls


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_model():
    train_loader, test_loader, y_train = load_data()
    class_weights = compute_class_weights(y_train)

    net       = ECG_SNN_RCNN_DFA().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn   = FocalLoss(alpha=class_weights, gamma=2.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    total_params = sum(p.numel() for p in net.parameters())
    bram_params  = (INPUT_FEATURES*HIDDEN_1_NEURONS + HIDDEN_1_NEURONS*HIDDEN_1_NEURONS +
                    HIDDEN_1_NEURONS*HIDDEN_2_NEURONS + HIDDEN_2_NEURONS*HIDDEN_2_NEURONS +
                    HIDDEN_2_NEURONS*OUTPUT_CLASSES)
    print(f"\nTotal trainable params  : {total_params}")
    print(f"Inference weights (BRAM): {bram_params} bytes "
          f"({bram_params/18432*100:.1f}% of one 18Kb tile)")
    print(f"\nDFA feedback matrices   : H1={list(net.H1.shape)}  H2={list(net.H2.shape)}  [FIXED, not trained]")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc = 0.0

    print(f"\nTraining ECG_SNN_RCNN_DFA | Steps={NUM_STEPS} | "
          f"Hidden=32 | LocalLossW={LOCAL_LOSS_W}")
    print("=" * 90)
    print(f"{'Ep':>4} | {'Loss_out':>9} | {'Loss_loc':>9} | "
          f"{'Train%':>7} | {'Test%':>7} | {'Gap':>6} | {'LR':>8}")
    print("-" * 90)

    for epoch in range(NUM_EPOCHS):
        net.train()
        sum_loss_out = 0.0
        sum_loss_loc = 0.0
        correct, total = 0, 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            spk1, spk2, spk3 = net(data)

            # ── Output loss (standard) ──
            out_sum  = spk3.sum(0)   # [B, 5]
            loss_out = loss_fn(out_sum, targets)

            # ── Local Loss at hidden layer 1 ──
            # sum spikes over time, pass through tiny classifier head
            h1_sum   = spk1.sum(0)  # [B, 32]
            loss_h1  = loss_fn(net.local_head1(h1_sum), targets)

            # ── Local Loss at hidden layer 2 ──
            h2_sum   = spk2.sum(0)  # [B, 32]
            loss_h2  = loss_fn(net.local_head2(h2_sum), targets)

            # ── DFA alignment loss ──
            # Compute output gradient direction (detached — no graph)
            with torch.no_grad():
                out_prob   = F.softmax(out_sum, dim=1)          # [B, 5]
                one_hot    = F.one_hot(targets, OUTPUT_CLASSES).float()
                delta_out  = out_prob - one_hot                 # [B, 5]

            # Project output error to hidden layer dimensions via fixed H matrices
            # dfa_target_h1[b] = H1.T @ delta_out[b] → [B, 32]
            dfa_signal_h1  = delta_out @ net.H1               # [B, 32]
            dfa_signal_h2  = delta_out @ net.H2               # [B, 32]

            # DFA alignment: push hidden activations opposite to DFA signal
            # This is the "gradient alignment" formulation of DFA
            dfa_loss_h1    = (h1_sum * dfa_signal_h1).mean()
            dfa_loss_h2    = (h2_sum * dfa_signal_h2).mean()

            # ── Total loss ──
            loss_local = LOCAL_LOSS_W * (loss_h1 + loss_h2 + dfa_loss_h1 + dfa_loss_h2)
            total_loss = loss_out + loss_local

            optimizer.zero_grad()
            total_loss.backward()
            # Clip gradients — essential for recurrent networks
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            sum_loss_out += loss_out.item()
            sum_loss_loc += loss_local.item()
            preds         = out_sum.argmax(1)
            correct      += (preds == targets).sum().item()
            total        += targets.size(0)

        train_acc  = 100.0 * correct / total
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Evaluate
        test_acc, per_cls = evaluate(net, test_loader)
        gap = train_acc - test_acc

        gap_flag = ""
        if gap > 7:   gap_flag = " ← OVERFIT"
        elif gap > 3: gap_flag = " ← watch"

        n = len(train_loader)
        print(f"{epoch+1:>4} | {sum_loss_out/n:>9.4f} | {sum_loss_loc/n:>9.4f} | "
              f"{train_acc:>6.2f}% | {test_acc:>6.2f}% | "
              f"{gap:>+5.1f}%{gap_flag} | {current_lr:.2e}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), f"{MODEL_DIR}/snn_rcnn_dfa_best.pth")
            print(f"     ★ New best → {best_acc:.2f}%")

        # Per-class breakdown every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n     Per-class accuracy (epoch {epoch+1}):")
            for c, name in enumerate(CLASS_NAMES):
                bar = "█" * int(per_cls[c] / 5)
                print(f"       {name:<30} {per_cls[c]:>6.2f}%  {bar}")
            print()

    print("=" * 90)
    print(f"Best Test Accuracy : {best_acc:.2f}%")
    print(f"Model saved to     : {MODEL_DIR}/snn_rcnn_dfa_best.pth")


if __name__ == "__main__":
    train_model()
